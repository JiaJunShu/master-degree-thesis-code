import pandas as pd

#C:\Users\Administrator\Documents\WeChat Files\wxid_r8bezitejwk621\FileStorage\File\2024-10\data\dataoutFinPhi0.csv
#C:\Users\Administrator\Documents\WeChat Files\wxid_r8bezitejwk621\FileStorage\File\2024-10\dataout20241007\dataoutFinPhi0.csv
file_path = r'C:\Users\Administrator\Documents\WeChat Files\wxid_r8bezitejwk621\FileStorage\File\2024-10\data\dataoutFinPhi350.csv'
# 根据检测到的编码读取 CSV 文件
try:
    df = pd.read_csv(file_path, encoding='utf-8')  # 尝试 UTF-8
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # 尝试 ISO-8859-1
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')  # 尝试 GBK

# 打印出 DataFrame 的前几行和列名
print("DataFrame 的前5行：")
print(df.head())  # 打印前五行
print("DataFrame 的列名：")
print(df.columns)  # 打印所有列名

# 检查列的总数
print("DataFrame 的列数：", df.shape[1])

# 创建一个新的 DataFrame 并按照新的列顺序添加数据
new_df = pd.DataFrame()

# 删除 RE 列中大于 1900000 的行
#df = df[df['RE'] <= 1900000]

# 使用 iloc 获取列，并确保列索引正确
if df.shape[1] > 29:  # 确保至少有30列
    new_df[0] = df.iloc[:, 3]   # 第4列 'MACH'
    new_df[1] = df.iloc[:, 4]   # 第5列 'RE'
    new_df[2] = df.iloc[:, 5]   # 第6列 'BETA'
    new_df[3] = df.iloc[:, 22]  # 第20列 'ALPHA'

    new_df[4] = df.iloc[:, 24]  # 第21列 'CM'
    new_df[5] = df.iloc[:, 38]  # 第22列 'CMQ'
    new_df[6] = df.iloc[:, 29]  # 第26列 'cln'
    new_df[7] = df.iloc[:, 89]  # 第27列 'clnr'
    new_df[8] = df.iloc[:, 93]  # 第27列 'clLP'
    new_df[9] = df.iloc[:, 28]  # 第28列 'CY'
    new_df[10] = df.iloc[:, 31]  # 第29列 'CL'
    new_df[11] = df.iloc[:, 32]  # 第29列 'CD'
else:
    print("Error: CSV文件的列数不足以满足要求。")

# 保存新的CSV文件
new_df.to_csv('output_file.csv', index=False)
