import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import joblib
import math
import torch.nn.functional as F
from dask.array import arctan

m=15 #炮弹质量
m_f = 0.11#翼筒质量
C_d=0.1 #阻力系数
C_l=0 #升力系数
C_y=0 #侧力系数


g=9.8
length = 0.12 #参考长度 气动弦长
diameter = 0.12 #直径
d=0.26#弹体质心和翼筒质心的距离
S = diameter * diameter * np.pi / 4 #面积

I_dx = 0.023898
I_dy = 0.307789
I_dz = 0.307789

I_fx = 96.22e-6
I_fy = 61.243e-6
I_fz = 61.243e-6

#加载模型
scaler_3round = joblib.load('scaler_3round.pkl')

# 定义与训练时相同的模型

# 定义全连接神经网络
class model_3round(nn.Module):
    def __init__(self):
        super(model_3round, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        # 第一个隐藏层
        self.fc2 = nn.Linear(32, 64)
        # 新增的隐藏层
        self.fc3 = nn.Linear(64, 32)
        # 第二个隐藏层
        self.fc4 = nn.Linear(32, 16)
        # 输出层
        self.fc5 = nn.Linear(16, 6)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return x


# 创建模型实例
model3 = model_3round()

# 加载模型的状态字典
model3.load_state_dict(torch.load('model_3round.pth'))

# 切换到评估模式
model3.eval()

class Missile:
    def __init__(self, position_x=0, position_y=0, position_z=0, v_xk=0, theta_a=0, psi_v=0,phi = 0,theta = 0,psi=0,omega_yb = 0,omega_zb = 0,omega_xb = 0,omega_fxb = 0,r_f=0,k3=0.75,k4=15,wind=None):
        #以下是位置的内容
        self.position_x = position_x  # x坐标
        self.position_y = position_y  # y坐标
        self.position_z = position_z  # z坐标
        self.v_xk = v_xk  # 速度大小
        self.theta_a = theta_a  # 高度角（俯仰角）
        self.psi_v = psi_v  # 侧滑角
        self.v_xg=v_xk*np.cos(theta_a)*np.cos(psi_v)
        self.v_yg=v_xk*np.cos(theta_a)*np.sin(psi_v)
        self.v_zg=-v_xk*np.sin(theta_a)

        #以下是姿态的内容
        self.phi = phi  # 弹体滚转角
        self.theta = theta  # 弹体俯仰角
        self.psi = psi  # 弹体偏航角
        self.r_f = r_f # 翼筒滚转角

        self.beta = np.arcsin(np.cos(self.theta_a) * (
                    np.sin(self.phi) * np.sin(self.theta) * np.cos(self.psi - self.psi_v) - np.cos(self.phi) * np.sin(
                self.psi - self.psi_v)) - np.sin(self.theta_a) * np.sin(self.phi) * np.cos(self.theta))  # 侧滑攻角
        self.alpha = np.arcsin((np.cos(self.theta_a) * (
                    np.cos(self.phi) * np.sin(self.theta) * np.cos(self.psi - self.psi_v) + np.sin(self.phi) * np.sin(
                self.psi - self.psi_v)) - np.sin(self.theta_a) * np.cos(self.phi) * np.cos(self.theta)) / np.cos(
            self.beta))  # 高低攻角
        #self.phi_w=np.arcsin((np.sin(self.theta_a)*(np.sin(self.phi)*np.sin(self.theta)*np.cos(self.psi - self.psi_v)-np.cos(self.psi) * np.sin(self.psi - self.psi_v))+np.cos(self.theta_a) * np.sin(self.psi) * np.cos(self.theta))/np.cos(self.beta))

        self.omega_yb = omega_yb  # 绕y轴角速度
        self.omega_zb = omega_zb  # 绕z轴角速度
        self.omega_xb = omega_xb  # 绕x轴角速度# 弹体滚转角速度
        self.omega_fxb = omega_fxb  # 翼筒滚转角速度

        self.Ma = self.v_xk/(0.0038*self.position_z+340.29) #马赫
        self.Re = (0.000112*self.position_z+1.225)*self.v_xk *0.12*100000/ (0.0000316*self.position_z+1.78941) #雷诺  算了一下导弹雷诺数一般是1.9*10六次方300v 到1.2*10六次方200v

        self.q=0.5*(0.000112*self.position_z+1.225)*self.v_xk*self.v_xk#动压
        self.D = C_d * self.q * S
        self.C = C_y * self.q * S
        self.L = C_l * self.q * S

        self.f_xg=0
        self.f_yg = 0
        self.f_zg = 0

        self.Mly=0
        self.Mlz = 0
        self.Mx = 0
        self.My = 0
        self.Mz = 0
        self.Te=0

        self.integral=0
        self.previous_error=0

        self.current_time = 0

        self.state2 = 0 #最高点起控信号
        self.r_fc=0 #期望翼筒滚转角
        self.r_fc_no_gratitude=0#无重力期待翼筒滚转角，用于对重力补偿做对比

        self.q2=0
        self.dq2dt=0
        self.tt = 0#最高点启控后开始计时

        self.wind=wind#风速

        self.k4=k4
        self.k3=k3

    #符号函数
    @staticmethod
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    #内环滑膜控制器
    def slide_mode_controller(self, desired_roll_angle, c1, c2):
        self.s1 = c1 * (self.r_f - desired_roll_angle) + c2 * self.omega_fxb
        # 计算扭矩 额定41 堵转265
        self.Te = (-0.8*self.sign(self.s1) - c1 * (self.omega_fxb + np.tan(self.theta) * (
                    self.omega_yb * np.sin(self.phi) + self.omega_zb * np.cos(self.phi)))) * I_fx / c2

    def pid_controller(self, desired_roll_angle, kp, ki, kd, dt):
        error = desired_roll_angle - self.r_f
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        # 计算扭矩
        self.Te = kp * error+ ki * self.integral + kd * derivative

        # 更新previous_error
        self.previous_error = error

    #角度归一化到-180,180
    def normalize_angle_radians(self, angle):
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle
    #求攻角和速度
    def attack_speed(self):

        self.v_xk=np.sqrt(self.v_xg*self.v_xg+self.v_yg*self.v_yg+self.v_zg*self.v_zg)

        self.theta_a=np.arcsin(-self.v_zg/self.v_xk)
        self.psi_v=np.arctan(self.v_yg/self.v_xg)

        self.beta = np.arcsin(np.cos(self.theta_a) * (np.sin(self.phi) * np.sin(self.theta) * np.cos(self.psi - self.psi_v) - np.cos(self.phi) * np.sin(self.psi - self.psi_v)) - np.sin(self.theta_a) * np.sin(self.phi) * np.cos(self.theta))  # 侧滑攻角
        self.alpha = np.arcsin((np.cos(self.theta_a)*(np.cos(self.phi)*np.sin(self.theta)*np.cos(self.psi-self.psi_v)+np.sin(self.phi)*np.sin(self.psi-self.psi_v))-np.sin(self.theta_a)*np.cos(self.phi)*np.cos(self.theta))/np.cos(self.beta))  # 高低攻角
        #相对速度滚转角
        '''
        #self.phi_w = np.arcsin((np.sin(self.theta_a) * (np.sin(self.phi) * np.sin(self.theta) * np.cos(self.psi - self.psi_v) - np.cos(self.phi) * np.sin(self.psi - self.psi_v)) + np.cos(self.theta_a) * np.sin(self.phi) * np.cos(self.theta)) / np.cos(self.beta))
        '''
    #求力
    def force(self):
        #先求气动系数
        # 假设 self.Ma, self.Re, self.alpha, self.beta 这些属性已经被定义

        new_data_scaled = scaler_3round.transform([[self.Ma, self.Re,self.beta*180/np.pi,self.alpha*180/np.pi,self.normalize_angle_radians(self.r_f)*180/np.pi]])  # 直接使用已经拟合好的 scaler
        # 将标准化后的数据转换为 PyTorch 张量
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        # 使用模型进行预测
        with torch.no_grad():  # 在评估时不需要计算梯度
            prediction = model3(new_data_tensor)
        # 先求气动系数
        C_n = prediction[0, 0]  # 法向力系数
        C_a  = prediction[0, 2]  # 轴向力系数
        C_y = prediction[0, 3]  # 侧力系数


        A_datcom = C_a * self.q * S  #  轴向力 都在弹体坐标系
        Y_datcom = C_y * self.q * S  # 侧力
        N_datcom = C_n * self.q * S # 法向力


        #这三个要从弹体转到地面
        Z=np.sin(self.phi)*np.sin(self.theta)*np.cos(self.psi)-np.cos(self.phi)*np.sin(self.psi)
        X=np.sin(self.phi)*np.sin(self.theta)*np.sin(self.psi)+np.cos(self.phi)*np.cos(self.psi)
        Y=np.cos(self.phi)*np.sin(self.theta)*np.cos(self.psi)+np.sin(self.phi)*np.sin(self.psi)
        Q=np.cos(self.phi)*np.sin(self.theta)*np.sin(self.psi)-np.sin(self.phi)*np.cos(self.psi)

        self.D=np.cos(self.theta)*np.cos(self.psi)*(-A_datcom)+Z*Y_datcom+Y*(-N_datcom)
        self.C=np.cos(self.theta)*np.sin(self.psi)*(-A_datcom)+X*Y_datcom+Q*(-N_datcom)
        self.L =-np.sin(self.theta)*(-A_datcom)+np.sin(self.phi)*np.cos(self.theta)*Y_datcom+np.cos(self.phi)*np.cos(self.theta)*(-N_datcom)


        '''
        self.D =-D
        self.C =C*np.cos(self.phi_w)+L*np.sin(self.phi_w)
        self.L =C*np.sin(self.phi_w)-L*np.cos(self.phi_w)
        '''

        self.f_xg = self.D
        self.f_yg = self.C
        self.f_zg = self.L + (m + m_f) * g


        if self.position_z<-1130:
            self.f_xg = self.D+self.wind
    def get_Te_method(self): #这个函数求Te
        if  self.state2 == 1:
            self.tt+=0.001

        self.slide_mode_controller(self.r_fc, 0.061, 0.00005)

        # 到最高点之前翼筒滚转角一直转
        if self.v_zg < 0:
             self.Te = I_fx * (50*2*np.pi - self.omega_fxb) #50转/s

        #到最高点以后翼筒滚转角和滚转角归一化一下。这很重要，不然结果非常不理想
        if abs(self.v_zg)<0.01:
             self.r_f=self.normalize_angle_radians(self.r_f)
             self.phi = 0#self.normalize_angle_radians(self.phi)
             self.state2 = 1#最高点启控制信号


    def get_r_fc_method2(self, x_pre, y_pre):  # 方法二这个函数先求r_fc

        x_target_position =4320
        y_target_position = 0

        detal_x=x_target_position-x_pre
        detal_y=y_target_position-y_pre

        if detal_x>0 and abs(detal_y)<=1:
            self.r_fc =0
        elif detal_x>0 and detal_y>0:
            self.r_fc = 1 / (1 + np.exp(detal_x/self.k4))*np.arctan(detal_y / detal_x)
        elif abs(detal_x)<=1 and detal_y>0:
            self.r_fc = 1 / (1 + np.exp(detal_x/self.k4))*np.pi/2
        elif detal_x<0 and detal_y>0:
            self.r_fc = 1 / (1 + np.exp(detal_x/self.k4))*(np.pi+np.arctan(detal_y / detal_x))
        elif detal_x<0 and abs(detal_y)<=1:
            self.r_fc = np.pi
        elif detal_x>0 and detal_y<0:
            self.r_fc = 1 / (1 + np.exp(detal_x/self.k4))*np.arctan(detal_y / detal_x)
        elif abs(detal_x)<=1 and detal_y<0:
            self.r_fc = -1 / (1 + np.exp(detal_x/self.k4))*np.pi/2
        elif detal_x<0 and detal_y<0:
            self.r_fc = 1 / (1 + np.exp(detal_x/self.k4))*(np.arctan(detal_y / detal_x)-np.pi)

    def get_r_fc_method1(self):  # 方法一这个函数求r_fc
        # 目标落点
        x_target_position = 4320
        y_target_position = 0

        r1=math.sqrt((x_target_position-self.position_x)*(x_target_position-self.position_x)+(y_target_position-self.position_y)*(y_target_position-self.position_y))
        q1=arctan((y_target_position-self.position_y)/(x_target_position-self.position_x))
        dq1dt=self.v_xk*np.cos(self.theta_a)*np.sin(q1-self.psi_v)/r1
        r_fcz=5000*dq1dt

        r2 = math.sqrt( r1*r1+self.position_z*self.position_z)
        q2 = arctan(self.position_z / r1 )
        self.dq2dt = self.v_xk * np.cos(q1-self.psi_v) * np.sin(q2-self.theta_a) / r2
        r_fcy = 4.2 * self.dq2dt-0.005*g/self.v_xk/np.cos(q1-self.psi_v)
        self.r_fcy=r_fcy

        if r_fcy > 0 and r_fcz > 0:
            self.r_fc = arctan(self.k3*r_fcz/(1-self.k3)/r_fcy)
        elif abs(r_fcy) ==0 and r_fcz > 0:
            self.r_fc = np.pi/2
        elif r_fcy < 0 and  r_fcz > 0:
            self.r_fc = np.pi + arctan(self.k3*r_fcz/(1-self.k3)/r_fcy)
        elif r_fcy < 0 and r_fcz ==0:
            self.r_fc = np.pi
        elif r_fcy > 0 and r_fcz <0:
            self.r_fc = arctan(self.k3*r_fcz/(1-self.k3)/r_fcy)
        elif r_fcy == 0 and r_fcz < 0:
            self.r_fc = -np.pi / 2
        elif r_fcy < 0 and r_fcz < 0:
            self.r_fc = arctan(self.k3*r_fcz/(1-self.k3)/r_fcy)- np.pi

        #以下探究无重力补偿
        r_fcy_no_gratitude = 4.2 * self.dq2dt
        self.r_fcy_no_gratitude = r_fcy_no_gratitude
        self.q2=q2
        if r_fcy_no_gratitude > 0 and r_fcz > 0:
            self.r_fc_no_gratitude = arctan(self.k3*r_fcz/(1-self.k3)/r_fcy_no_gratitude)
        elif abs(r_fcy_no_gratitude) ==0 and r_fcz > 0:
            self.r_fc_no_gratitude = np.pi/2
        elif r_fcy_no_gratitude < 0 and  r_fcz > 0:
            self.r_fc_no_gratitude = np.pi + arctan(self.k3*r_fcz/(1-self.k3)/r_fcy_no_gratitude)
        elif r_fcy_no_gratitude < 0 and r_fcz ==0:
            self.r_fc_no_gratitude = np.pi
        elif r_fcy_no_gratitude > 0 and r_fcz <0:
            self.r_fc_no_gratitude = arctan(self.k3*r_fcz/(1-self.k3)/r_fcy_no_gratitude)
        elif r_fcy_no_gratitude == 0 and r_fcz < 0:
            self.r_fc_no_gratitude = -np.pi / 2
        elif r_fcy_no_gratitude < 0 and r_fcz < 0:
            self.r_fc_no_gratitude = arctan(self.k3*r_fcz/(1-self.k3)/r_fcy_no_gratitude)- np.pi

    def torque(self):
        self.Ma=self.v_xk/(0.0038*self.position_z+340.29)#  0.3-0.9
        self.Re = (0.000112*self.position_z+1.225)*self.v_xk *0.12*100000/ (0.0000316*self.position_z+1.78941)  #1.9-1.1 0.12是纵向长度  地面密度1.241
        #print(f"马赫数 (psi): {self.Ma} ")
        #print(f"雷诺数 (psi): {self.Re} ")
        # 使用之前训练时的 scaler 进行标准化

        new_data_scaled = scaler_3round.transform([[self.Ma, self.Re,self.beta*180/np.pi,self.alpha*180/np.pi,self.normalize_angle_radians(self.r_f)*180/np.pi]])  # 直接使用已经拟合好的 scaler
        # 将标准化后的数据转换为 PyTorch 张量
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        # 使用模型进行预测
        with torch.no_grad():  # 在评估时不需要计算梯度
           prediction = model3(new_data_tensor) #cm  cmq cln cll cn  ca cy clnr cllp
         # 先求气动系数
        mx = 0#+prediction[0, 8]*self.omega_xb*length/2/self.v_xk #滚转力矩系数cll  cllp
        my = prediction[0, 1]+prediction[0, 5]*self.omega_yb*length/2/self.v_xk  # 俯仰力矩系数  cm cmq
        mz = prediction[0, 4]#+prediction[0, 7]*self.omega_zb*length/2/self.v_xk   # 偏航力矩系数   cln clnr

        self.q = 0.5 * (0.000112*self.position_z+1.225) * self.v_xk * self.v_xk

        self.Mx = mx * self.q * S * length
        self.My = my *self.q * S * length
        self.Mz = mz *self.q * S * length


    def dynamics(self, t, y):
         """定义导弹的动力学方程，质心运动方程"""
         position_x, position_y, position_z, v_xg, v_yg, v_zg = y

         # 根据导弹状态计算微分方程，根据需要自行调整
         dxdt =v_xg  # x方向速度
         dydt =v_yg  # y方向速度
         dzdt = v_zg  # z方向速度

         dv_xgdt =self.f_xg/(m+m_f)  # 速度x方向的变化率
         dv_ygdt = self.f_yg / (m+m_f)   # 速度y方向变化率
         dv_zgdt= self.f_zg / (m+m_f)  # 速度z方向变化率

         return np.array([dxdt, dydt, dzdt, dv_xgdt, dv_ygdt, dv_zgdt])

    def attitude_dynamics(self, t, y):
         """定义新的动力学方程，旋转运动方程"""
         phi, theta, psi, omega_xb, omega_yb, omega_zb, omega_fxb, r_f = y

         # 这是翼筒的两个参数
         dr_f = omega_fxb + np.tan(theta) * (omega_yb * np.sin(phi) + omega_zb * np.cos(phi))
         domega_fxbdt = self.Te/ I_fx

         # 根据状态计算微分方程 后体
         dphidt = omega_xb + np.tan(theta) * (omega_yb * np.sin(phi) + omega_zb * np.cos(phi))
         dthetadt = omega_yb * np.cos(phi) - omega_zb * np.sin(phi)
         dpsidt = (omega_yb * np.sin(phi) + omega_zb * np.cos(phi)) / np.cos(theta)

         domega_xbdt = (self.Mx - self.Te) / I_dx
         domega_ybdt = ((I_dx - I_dz) * omega_zb * omega_xb + (I_fx - I_fz - d * d * m_f) * omega_zb * omega_fxb + self.My) / (I_dy + I_fz + d * d * m_f)
         domega_zbdt = ((I_dy - I_dx) * omega_xb * omega_yb + (I_fy + d * d * m_f - I_fx) * omega_fxb * omega_yb + self.Mz) / (I_dz + I_fz + d * d * m_f)
         #domega_xbdt = (self.Mx - self.Te) / I_dx
         #domega_ybdt = ((I_dx - I_dz) * omega_zb * omega_xb  + self.My) / I_dy
         #domega_zbdt = ((I_dy - I_dx) * omega_xb * omega_yb  + self.Mz) / I_dz

         return np.array([dphidt, dthetadt, dpsidt, domega_xbdt, domega_ybdt, domega_zbdt, domega_fxbdt, dr_f])

    def runge_kutta_update(self, h= 0.001):
         """使用四阶龙哥库塔方法更新导弹状态，质心"""
         # 当前状态
         self.current_time += h
         t = 0  # 当前时间（可忽略，只用作占位）
         y = np.array([self.position_x, self.position_y, self.position_z, self.v_xg, self.v_yg, self.v_zg])

         # 计算k1
         k1 = self.dynamics(t, y)

         # 计算k2
         k2 = self.dynamics(t + h / 2, y + h * k1 / 2)

         # 计算k3
         k3 = self.dynamics(t + h / 2, y + h * k2 / 2)

         # 计算k4
         k4 = self.dynamics(t + h, y + h * k3)

         # 更新状态
         y_next = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

         # 更新导弹状态
         self.position_x, self.position_y, self.position_z, self.v_xg, self.v_yg, self.v_zg = y_next

    def attitude_runge_kutta_update(self, h= 0.001):
         """使用四阶龙哥库塔方法更新导弹状态，旋转"""
         # 当前状态
         t = 0  # 当前时间（可忽略，只用作占位）
         y = np.array([self.phi, self.theta, self.psi, self.omega_xb, self.omega_yb, self.omega_zb,self.omega_fxb,self.r_f])

         # 计算k1
         k1 = self.attitude_dynamics(t, y)

         # 计算k2
         k2 = self.attitude_dynamics(t + h / 2, y + h * k1 / 2)

         # 计算k3
         k3 = self.attitude_dynamics(t + h / 2, y + h * k2 / 2)

         # 计算k4
         k4 = self.attitude_dynamics(t + h, y + h * k3)

         # 更新状态
         y_next = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

         # 更新导弹状态
         self.phi, self.theta, self.psi, self.omega_xb, self.omega_yb, self.omega_zb,self.omega_fxb,self.r_f = y_next

    def display_info(self):
         print("导弹信息:")
         print(f"速度: {self.v_xk}")
         print(f"坐标: ({self.position_x}, {self.position_y}, {self.position_z})")
         print(f"爬升角 (theta_a): {self.theta_a*180/np.pi} ")
         print(f"航迹方位角 (psi_v): {self.psi_v * 180 / np.pi} ")
         print(f"俯仰角 (theta): {self.theta*180/np.pi} ")
         print(f"偏航角 (psi): {self.psi * 180 / np.pi} ")
         #print(f"后体滚转角 (phi): {self.phi * 180 / np.pi} ")
         #print(f"翼筒滚转角 (r_f): {self.normalize_angle_radians(self.r_f) * 180 / np.pi} ")
         print(f"后体x轴角速度 (omega_xb): {self.omega_xb} ")
         print(f"前体x轴角速度 (omega_fxb): {self.omega_fxb} ")
         #print(f"后体y轴角速度 (omega_yb): {self.omega_yb} ")
         #print(f"后体z轴角速度 (omega_zb): {self.omega_zb} ")
         #print(f"滚转角: {self.psi_v}")
         #print(f"姿态俯仰角theta: {self.theta}ra
         print(f"攻角 (alpha): {self.alpha*180/np.pi} ")
         print(f"侧滑角 (beta): {self.beta*180/np.pi} ")

    def save_to_file(self):
         # 将 self.theta 的值写入到 output.txt 文件中
         with open('output.txt', 'a', encoding='utf-8') as f:
             #f.write(str(self.normalize_angle_radians(self.r_fc)*180/np.pi)+'\t')  # 将数字转换为字符串
             f.write(str(self.r_fc*180/np.pi) + '\t')
             f.write(str(self.phi*180/np.pi) + '\t')
             f.write(str(self.theta*180/np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.position_x)+'\t')  # 将数字转换为字符串
             f.write(str(self.position_y) + '\t')  # 将数字转换为字符串
             f.write(str(-self.position_z) + '\t')  # 将数字转换为字符串
             f.write(str(self.current_time) + '\n')  # 将数字转换为字符串
             f.write(str(self.psi * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.omega_fxb * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.omega_xb * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.omega_yb * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.omega_zb * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.psi_v * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.theta_a * 180 / np.pi) + '\t')  # 将数字转换为字符串
             f.write(str(self.v_xk ) + '\n')  # 将数字转换为字符串

