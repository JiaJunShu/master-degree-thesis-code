import numpy as np

import torch
import torch.nn as nn
import joblib
import math
import torch.nn.functional as F
from dask.array import arctan

m=45#炮弹质量
m_f = 0.11#翼筒质量
C_d=0.1 #阻力系数
C_l=0 #升力系数
C_y=0 #侧力系数
mx=-0.75#力矩系数  这个也是我随便给的 需要训练出来
my=-0.0005#俯仰力矩系数
mz=0.01#偏航力矩系数
ml = 0.1#升力舵力矩系数  这个也是我随便给的 需要训练出来 这个最好小一点  小于0.3

g=9.8
PI=3.1415926
length = 0.12 #参考长度 气动弦长
diameter = 0.12 #直径
d=0.06#弹体质心和翼筒质心的距离
S = diameter * diameter * PI / 4 #面积
rho=1.293  #空气密度
I_dx = 0.023898
I_dy = 0.307789
I_dz = 0.307789

I_fx = 96.22e-6
I_fy = 61.243e-6
I_fz = 61.243e-6


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
        self.fc3 = nn.Linear(64, 64)
        # 第二个隐藏层
        self.fc4 = nn.Linear(64, 32)
        # 输出层
        self.fc5 = nn.Linear(32, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# 创建模型实例
model3 = model_3round()

# 加载模型的状态字典
model3.load_state_dict(torch.load('model_3round.pth'))

# 切换到评估模式
model3.eval()

class Missile:
    def __init__(self, position_x=0, position_y=0, position_z=0, v_xk=0, theta_a=0, psi_v=0,phi = 0,theta = 0,psi=0,omega_yb = 0,omega_zb = 0,omega_xb = 0):
        #以下是位置的内容
        self.position_x = position_x  # x坐标
        self.position_y = position_y  # y坐标
        self.position_z = position_z  # z坐标
        self.v_xk = v_xk  # 速度
        self.theta_a = theta_a  # 高度角（俯仰角）
        self.psi_v = psi_v  # 侧滑角
        self.alpha = theta-theta_a  # 高低 攻角
        self.beta = psi_v-psi  # 侧滑攻角
        #以下是姿态的内容
        self.phi = phi  # 弹体滚转角
        self.theta = theta  # 弹体俯仰角
        self.psi = psi  # 弹体偏航角
        self.r_f = 0 # 翼筒滚转角

        self.omega_yb = omega_yb  # 绕y轴角速度
        self.omega_zb = omega_zb  # 绕z轴角速度
        self.omega_xb = omega_xb  # 绕x轴角速度# 弹体滚转角速度
        self.omega_fxb = 0  # 翼筒滚转角速度

        self.Ma = 0.4 #马赫
        self.Re = 1500000 #雷诺  算了一下导弹雷诺数一般是1.9*10六次方300v 到1.2*10六次方200v

        self.q=0.5*rho*self.v_xk*self.v_xk
        self.D = C_d * self.q * S
        self.C = C_y * self.q * S
        self.L = C_l * self.q * S

        self.f_xk=0
        self.f_yk = 0
        self.f_zk = 0

        self.Mly=0
        self.Mlz = 0
        self.Mx = 0
        self.My = 0
        self.Mz = 0
        self.Te=0

        self.integral=0
        self.previous_error=0

        self.current_time = 0

        self.state=0
    @staticmethod
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def slide_mode_controller(self, desired_roll_angle, c1, c2):
        self.s1 = c1 * (self.r_f - desired_roll_angle) + c2 * self.omega_fxb
        # 计算扭矩
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

    def normalize_angle_radians(self, angle):
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle

    def attack(self):
        self.alpha = self.theta-self.theta_a  # 高低攻角
        self.beta = self.psi_v-self.psi  # 侧滑攻角
    def force(self):
        #先求气动系数
        # 假设 self.Ma, self.Re, self.alpha, self.beta 这些属性已经被定义
        self.r_f = self.normalize_angle_radians(self.r_f)
        #self.r_f = -np.pi
        new_data_scaled = scaler_3round.transform([[self.Ma, self.Re,self.beta*180/PI,self.alpha*180/PI,self.r_f*180/PI]])  # 直接使用已经拟合好的 scaler
        # 将标准化后的数据转换为 PyTorch 张量
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        # 使用模型进行预测
        with torch.no_grad():  # 在评估时不需要计算梯度
            prediction = model3(new_data_tensor)
        # 先求气动系数
        C_y = prediction[0, 5]  # 侧力系数
        C_l  = prediction[0, 6]  # 升力系数
        C_d = prediction[0, 7]  # 阻力系数

        self.C = C_y * self.q * S  # 侧力
        self.L = C_l * self.q * S  # 升力
        self.D = C_d * self.q * S # 阻力



        self.f_xk=-self.D-(m+m_f)*g*np.sin(self.theta_a)
        self.f_yk = self.C
        self.f_zk = -self.L+(m+m_f)*g*np.cos(self.theta_a)

    def torque(self):
        self.Ma=self.v_xk/(331.3+0.6*25)#20是空气温度  0.3-0.9
        self.Re = 1.241*self.v_xk *0.12*100000/ 1.81  #1.9-1.2 0.12是纵向长度
        #print(f"马赫数 (psi): {self.Ma} ")
        #print(f"雷诺数 (psi): {self.Re} ")
        # 使用之前训练时的 scaler 进行标准化
        # r_f限制在-180 180 之间
        self.r_f =self.normalize_angle_radians(self.r_f)
        '''  #self.r_f = -np.pi
        q=arctan((170-self.position_y)/(4800-self.position_x))
        r=math.sqrt((170-self.position_y)*(170-self.position_y)+(4800-self.position_x)*(4800-self.position_x))
        #jiaopinglv=math.sqrt(r)*100
        #if jiaopinglv < 80000:
         #   jiaopinglv = 80000
        #r_fc=70*(q-self.psi_v)+ 1.4*np.sin(self.current_time*jiaopinglv)
        #if r_fc>np.pi/2:
        #    r_fc=np.pi/2
        print(f"q: {q*180/PI} ")
        print(f"self.psi_v: {self.psi_v * 180 / PI} ")

        r_fc=2000*(self.v_xk*np.cos(self.theta_a)*np.sin(q-self.psi_v))/r
        if abs(r_fc)>np.pi/2:
            r_fc=np.pi/2
        self.slide_mode_controller(r_fc, 0.3, 0.005)
        if abs(q-self.psi_v)<np.pi*0.01/180:
            #self.Te+=0.035*np.sin(self.current_time*0.01)
            self.state=1
        if self.state==1:
            self.slide_mode_controller(np.pi*0.95*(-self.position_z/r)*np.sin(self.current_time*0.2), 0.3, 0.005)#current_time后面那个参数用0.2
        # 根据师兄滑膜控制方法来求力矩
        '''
        new_data_scaled = scaler_3round.transform([[self.Ma, self.Re,self.beta*180/PI,self.alpha*180/PI,self.r_f*180/PI]])  # 直接使用已经拟合好的 scaler
        # 将标准化后的数据转换为 PyTorch 张量
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

        # 使用模型进行预测
        with torch.no_grad():  # 在评估时不需要计算梯度
            prediction = model3(new_data_tensor) #cm cmq cln clnr cllp cy cl   cd
        # 先求气动系数
        mx = prediction[0, 4]*self.omega_xb #滚转力矩系数cllp
        my = prediction[0, 0]+prediction[0, 1]*self.omega_yb  # 俯仰力矩系数  cm cmq
        mz = prediction[0, 2]+prediction[0, 3]*self.omega_zb  # 偏航力矩系数   cln clnr

        self.q = 0.5 * rho * self.v_xk * self.v_xk

        self.Mx = mx * self.q * S * length
        self.My = my *self.q * S * length
        self.Mz = mz *self.q * S * length
        #根据师兄滑膜控制方法来求力矩
        #self.pid_controller(np.pi/6,0.0001,0,0,0.0001)


        #print(f"self.My: {self.My} ")
        #print(f"self.Mz: {self.Mz} ")
    def dynamics(self, t, y):
        """定义导弹的动力学方程"""
        position_x, position_y, position_z, v_xk, theta_a, psi_v = y

        # 根据导弹状态计算微分方程，根据需要自行调整
        dxdt = v_xk * np.cos(theta_a) * np.cos(psi_v)  # x方向速度
        dydt = v_xk * np.cos(theta_a) * np.sin(psi_v)  # y方向速度
        dzdt = -v_xk * np.sin(theta_a)  # z方向速度

        dv_xkdt =self.f_xk/(m+m_f)  # 速度的变化率，具体取决于应用场景
        dtheta_adt = -self.f_zk / (m+m_f) / v_xk  # 高度角变化率，具体取决于应用场景
        dpsi_vdt= self.f_yk / (m+m_f)/v_xk/np.cos(theta_a)  # 侧滑角变化率，具体取决于应用场景

        return np.array([dxdt, dydt, dzdt, dv_xkdt, dtheta_adt, dpsi_vdt])

    def attitude_dynamics(self, t, y):
        """定义新的动力学方程"""
        phi, theta, psi,omega_xb,omega_yb,omega_zb,omega_fxb,r_f = y

        # 这是翼筒的两个参数
        dr_f = omega_fxb + np.tan(theta) * (omega_yb * np.sin(phi) + omega_zb * np.cos(phi))
        domega_fxbdt = self.Te / I_fx

        # 根据状态计算微分方程
        dphidt = omega_xb + np.tan(theta) * (omega_yb * np.sin(phi) + omega_zb * np.cos(phi))
        dthetadt = omega_yb * np.cos(phi) - omega_zb * np.sin(phi)
        dpsidt = (omega_yb * np.sin(phi) + omega_zb * np.cos(phi)) / np.cos(theta)

        domega_xbdt=(self.Mx-self.Te)/I_dx
        domega_ybdt = ((I_dx - I_dz) * omega_zb * omega_xb +(I_fx - I_fz-d*d*m_f) * omega_zb * omega_fxb +self.My)/(I_dy+I_fz+d*d*m_f)
        domega_zbdt = ((I_dy - I_dx) * omega_xb * omega_yb +(I_fy+d*d*m_f - I_fx) * omega_fxb * omega_yb+self.Mz)/(I_dz+I_fz+d*d*m_f)

        return np.array([dphidt, dthetadt, dpsidt, domega_xbdt, domega_ybdt, domega_zbdt,domega_fxbdt,dr_f])

    def runge_kutta_update(self, h=0.002):
        """使用四阶龙哥库塔方法更新导弹状态"""
        # 当前状态
        self.current_time += h
        t = 0  # 当前时间（可忽略，只用作占位）
        y = np.array([self.position_x, self.position_y, self.position_z, self.v_xk, self.theta_a, self.psi_v])

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
        self.position_x, self.position_y, self.position_z, self.v_xk, self.theta_a, self.psi_v = y_next

    def attitude_runge_kutta_update(self, h=0.002):
        """使用四阶龙哥库塔方法更新导弹状态"""
        # 当前状态
        #self.current_time += h
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
        print(f"速度高度角 (theta_a): {self.theta_a*180/np.pi} ")
        print(f"姿态高度角 (theta): {self.theta*180/np.pi} ")
        print(f"速度侧滑角 (psi_v): {self.psi_v*180/np.pi} ")
        #print(f"滚转角: {self.psi_v}")
        #print(f"姿态俯仰角theta: {self.theta}ra
        print(f"姿态偏航角 (psi): {self.psi*180/np.pi} ")
        print(f"高低攻角 (alpha): {self.alpha*180/np.pi} ")
        print(f"侧滑攻角 (beta): {self.beta*180/np.pi} ")
        print(f"翼筒滚转角 (r_f): {self.r_f * 180 / np.pi} ")


        #print(f"俯仰角速度（omega_yb）: {self.omega_yb}rad")

        #print(f"偏航角速度 (psi): {self.omega_zb} rad")
        #print(f"偏航角速度 (psi): {self.omega_xb} rad")
    def save_to_file(self):
        # 将 self.theta 的值写入到 output.txt 文件中
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(str(self.r_f*180/np.pi)+'\t')  # 将数字转换为字符串
            f.write(str(self.psi*180/np.pi) + '\t')  # 将数字转换为字符串
            f.write(str(self.phi*180/np.pi) + '\t')  # 将数字转换为字符串
            f.write(str(self.position_x)+'\t')  # 将数字转换为字符串
            f.write(str(self.position_y) + '\t')  # 将数字转换为字符串
            f.write(str(-self.position_z) + '\t')  # 将数字转换为字符串
            f.write(str(self.current_time) + '\n')  # 将数字转换为字符串
# 示例使用
#missile = Missile(position_x=0, position_y=0, position_z=0, v_xk=10, theta_a=np.pi/6, psi_v=np.pi/4)
#for _ in range(100):  # 更新状态，循环100次
   # missile.runge_kutta_update()
    #missile.display_info()
