这是我的硕士毕业论文代码，用以对翼筒稳定弹进行仿真。 分为py代码和matlab代码,以下分别介绍。

一、py代码

平台：pycharm，需要安装pytorch框架，使用本机python3.10（base）环境

使用流程，必须先运行1和2得到神经网络代理模型：
1.zhenghe.py
作用：将datcom得到的数据整合成我想要的格式，以excel形式保存，用来作为数据集。
输入：file_path路径下的所有excel表格（共计73个，一个表代表一个不同翼筒滚转角），都是使用datcom拉出来的数据。
输出：一个csv文件“combined_output.csv”，从第一列开始分别代表3马赫 4雷诺 5beta 22alpha 翼筒滚转角 24cm  38cmq 29cln 23cn  25ca 28cy
ps：原始表格的翼筒滚转角度是0-360，我转化为了-180-180。

2.model3round.py
作用：使用第一步的训练集“combined_output.csv”训练一个bp神经网络。
输入：“combined_output.csv”，使用z-score方式对数据标准化。
输出：训练完成后，保存为'model_3round.pth'
ps：网络结构5输入6输出，学习率lr，训练轮次num_epochs,批次大小batch_size可修改，激活函数sigmoid。

3.model3predict.py(可选)[py仿真代码](py%E4%BB%BF%E7%9C%9F%E4%BB%A3%E7%A0%81)
作用：测试第二步的模型的准确性
输入：改变new_data作为自定义的输[py仿真代码](py%E4%BB%BF%E7%9C%9F%E4%BB%A3%E7%A0%81)入
输出：prediction是该模型预测的输出
ps：将prediction和combined_output.csv里面的数据进行对比可以查看准确与否。

4.main.py
作用：仿真程序的主函数
输入： 第21行：for w in wind_values，改变风速
      第21行：for x in x_values,初始条件psi和psi_v修改为x * np.pi / 180产生随机初始发射偏角
      第21行：for y in y_values,初始条件r_f修改为y * np.pi / 180产生随机初始翼筒滚转角
      第21行：for z in z_values，初始条件theta和theta_a修改为z * np.pi / 180产生随机初始射角
      其他初始条件可自行更改
      missle.py里面可修改目标落点x_target_position、y_target_position
      方法一和方法二可按照需要自行注释
输出：mtkl_method1_wind_zong_v220_4320,0.txt，代表方法一、初速220，改变纵向风速，目标落点4320,0所得到的蒙脱卡洛仿真结果。
     前两列代表仿真落点，第三列代表初始条件（修改第80行选择是否记录初始条件）
ps：第21行，同时改变x，y可以这样写for x,y in zip(x_values,y_values):

5.zhengding_k3_k4_error.py
作用：整定k3和k4
输入： 和main.py类似，自行改变初始条件
输出： 保存为'zhengdingk3_error_k3.txt'，前两列代表实际落点，后两列代表初始发射偏角和初始翼筒滚转角。这个文件需要经过整理，得到k3error.xlsx
ps：我统计xlsx文件的时候直接询问的ai，xlsx文件在论文中包含三个重要指标。x方向误差平均值、y方向误差平均值、总体误差平均值

6.get_impact_point_range.py
作用：得到炮弹在指定初速下的调控范围
输入： v_xk初速
输出： luodianfanwei_v250.txt代表初速250条件下的落点范围，第三列数据代表翼筒滚转角保持的角度
ps：

二、matlab代码

平台：matlab

1.bianxishuhanshu.m
作用：绘制不同系数k4对sigmoid函数的影响
输入：无
输出：k4变化对sigmoid函数的变化
ps：

2.cehuajiao_error.m
作用：绘制两种方法下，初始侧滑角改变对落点误差的影响
输入：mtkl_method1_cehuajiao_v220_4320,0.txt和mtkl_method2_cehuajiao_v220_4320,0.txt
输出：横坐标为初始侧滑角，纵坐标为误差
ps：

3.DRAW3D.m
作用：绘制弹道3d图
输入：output.txt（单次仿真的所有状态变量）
输出：弹道图
ps：可以得到xoy xoz平面图，提取的数据顺序具体要看missle.py里面的save_to_file函数的保存顺序，下面涉及到output.txt的也同样如此

4.draw_alpha_beta.m
作用：绘制角度随着时间变化图
输入：output.txt（单次仿真的所有状态变量）
输出：横坐标时间，纵坐标output的前三列，分开绘制
ps：

5.draw_qiwanggunzhuanjiao.m
作用：绘制期望翼筒滚转角随着时间变化图
输入：output.txt（单次仿真的所有状态变量）
输出：横坐标时间，纵坐标期望翼筒滚转角
ps：从15秒开始启控，所以程序里面设置了startIndex = 15062。15s这个数值要看具体仿真情况

6.draw_sudu_jiaodu.m
作用：绘制速度的两个角度随着时间变化图
输入：output.txt（单次仿真的所有状态变量）
输出：横坐标时间，纵坐标航迹倾斜角、航迹方位角，分开绘制
ps：

7.draw_wukongsudu.m
作用：绘制速度大小随着时间变化图
输入：output.txt（单次仿真的所有状态变量）
输出：横坐标时间，纵坐标速度大小
ps：

8.draw_zitai_8ge.m
作用：绘制角速度和角速度大小随着时间变化图
输入：output.txt（单次仿真的所有状态变量）
输出：横坐标时间，纵坐标4个角速度和4个角速度大小（共8个）
ps：

9.drawxy.m
作用：绘制落点范围
输入：round2_luodianfanwei_v220.txt落点范围数据
输出：一个落点范围x代表射程，y代表侧偏
ps：

10.jihuohanshu.m
作用：绘制激活函数
输入：
输出：sigmoid、relu、tanh函数
ps：

11.k3_error.m
作用：绘制error随着k3变化
输入：k3error.xlsx
输出：x方向、y方向、总体误差随k3变化
ps：

12.k4_error.m
作用：绘制error随着k4变化
输入：k4error.xlsx
输出：x方向、y方向、总体误差随k3变化
ps：

13.loss_decrease.m
作用：绘制损失随着训练轮次的下降情况
输入：loss_values1、manual_loss_values分别代表训练集和测试集
输出：x方向、y方向、总体误差随k3变化
ps：loss_values1、manual_loss_values是我自己写的，实际上得在训练的时候记录下来，规律一般都是下降

14.mengtuokaluo.m
作用：绘制蒙特卡洛仿真的散点图
输入：mtkl_method1_v280_6275,0.txt、mtkl_method2_v280_6275,0.txt，就是蒙特卡洛仿真的图
输出：落点分布情况
ps：根据数据里面的所有落点自行修改范围和目标点

15.method1_buchangxishu.m
作用：绘制补偿比例虽补偿前角度的变化
输入：x补偿前的角度，y补偿后的角度
输出：y/x随着x的变化情况
ps：用来反映补偿项的作用

16.dandao_2d.m
作用：画出弹道的平面图
输入：一次弹道仿真output.txt文件
输出：平面中的弹道轨迹
ps：根据需要修改读取第4列5列（xoy平面），或者第4列6列（xoz平面）