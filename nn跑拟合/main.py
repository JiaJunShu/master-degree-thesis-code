import numpy as np
from missle import Missile
import copy
import math
import random


def main():

                missile1 = Missile(v_xk=220, phi=0, theta=45 * np.pi / 180, psi=0 * np.pi / 180, position_x=0,
                                   position_y=0,
                                   position_z=0, theta_a=45 * np.pi / 180, psi_v=0 * np.pi / 180, omega_xb=0,
                                   omega_fxb=0,
                                   r_f=0 * np.pi / 180,k3=0.75 )
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
                    #if missile1.tt % 1 <= 0.0015 and missile1.state2 == 1:
                    missile1.get_r_fc_method1()

                    missile1.get_Te_method()
                    missile1.force()
                    missile1.attitude_runge_kutta_update()
                    missile1.runge_kutta_update()
                    missile1.display_info()
                    missile1.save_to_file()






if __name__ == "__main__":
    main()  # 调用主函数

