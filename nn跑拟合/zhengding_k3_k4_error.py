import numpy as np
from missle import Missile
import copy
import math
import random


def main():
        # 实例化Missle类，初始化参数 #速度别超过150 220
        x_pre = 1000
        y_pre = 1000

        # 生成两个随机序列
        x_values = [random.uniform(-0.5, 0.5) for _ in range(30)]
        y_values = [random.uniform(-180, 180) for _ in range(30)]
        # 生成一个包含 36 个从 0 到 1 的顺序序列
        sequence = np.arange(0.05, 1, 0.05)

        temp = 0
        # 清空文件
        open('zhengdingk3_error_k3.txt', 'w').close()
        for z in sequence:
            with open('output.txt', 'r', encoding='utf-8') as infile, open('zhengdingk3_error_k3.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(f"{z}\n")
            for x, y in zip(x_values, y_values):
                missile1 = Missile(v_xk=220, phi=0, theta=45 * np.pi / 180, psi=x * np.pi / 180, position_x=0,
                                   position_y=0,
                                   position_z=0, theta_a=45 * np.pi / 180, psi_v=x * np.pi / 180, omega_xb=0,
                                   omega_fxb=0,
                                   r_f=y * np.pi / 180,k3=z )
                # 清空文件
                open('output.txt', 'w').close()

                missile2 = None
                temp = 0
                while missile1.position_z <= 0:  #
                    missile1.attack_speed()
                    '''          # 方法二
                    if missile1.state2 == 1:
                        if missile1.tt % 1 <= 0.0015:
                            missile2 = copy.copy(missile1)

                            while missile2 is not None and missile2.position_z <= 0:
                                missile2.attack_speed()
                                missile2.torque()
                                print(f"预测坐标: ({missile2.position_x}, {missile2.position_y})")
                                missile2.slide_mode_controller(missile1.r_f, 0.061, 0.00005)  # 这一步主要是求力矩Te
                                missile2.force()
                                missile2.attitude_runge_kutta_update()
                                missile2.runge_kutta_update()
                                x_pre = missile2.position_x
                                y_pre = missile2.position_y

                            missile1.get_r_fc_method2(x_pre, y_pre)  # 求一下missile1.r_fc


                    else:
                        # 可能记录日志或输出信息来检查为何`state`不为`1`
                        print("Missile2未能初始化，因为missile1.state2不是1")
'''
                    missile1.torque()
                    # 方法一
                    if missile1.tt % 1 <= 0.0015 and missile1.state2 == 1:
                        missile1.get_r_fc_method1()

                    missile1.get_Te_method()
                    missile1.force()
                    missile1.attitude_runge_kutta_update()
                    missile1.runge_kutta_update()
                    missile1.display_info()
                    missile1.save_to_file()
                with open('output.txt', 'r', encoding='utf-8') as infile, open('zhengdingk3_error_k3.txt', 'a',
                                                                               encoding='utf-8') as outfile:
                    # 读取所有行
                    lines = infile.readlines()
                    # 获取最后一行
                    last_line = lines[-1]
                    # 按空格分隔并提取第4和第5列（索引为3和4）
                    columns = last_line.split()
                    if len(columns) >= 5:  # 确保至少有5列
                        data_to_copy = f"{columns[3]} {columns[4]} "
                        # 写入目标文件
                        outfile.write(data_to_copy + '\t')
                        outfile.write(str(x) + '\t')  # 在后面添加 i
                        outfile.write(str(y) + '\n')


if __name__ == "__main__":
    main()  # 调用主函数

