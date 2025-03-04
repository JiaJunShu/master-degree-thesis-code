import pandas as pd
import glob
import os
import re

# 文件路径
file_path = r'C:\Users\Administrator\Documents\WeChat Files\wxid_r8bezitejwk621\FileStorage\File\2024-10\data'

# 要选择的列的索引
column_indices = [3, 4, 5, 22, 24, 38, 29, 23, 25, 28] #3ma 4re 5beta 22alpha 24cm  38cmq 29cln 30cll 23cn  25ca 28cy 89clnr 93cllp

# 正则表达式匹配数字
pattern = r'dataoutFinPhi(\d{1,3})\.csv'

# 获取所有CSV文件的路径
all_files = glob.glob(os.path.join(file_path, 'dataoutFinPhi*.csv'))

# 初始化一个空的DataFrame
combined_df = pd.DataFrame()

for file in all_files:
    # 提取文件名中的数字
    match = re.search(pattern, file)
    if match:
        num = int(match.group(1))

        # 读取CSV文件
        df = pd.read_csv(file, usecols=column_indices)

        # 插入提取的数字作为新列
        df.insert(4, 'FileNumber', num)

        # 将FileNumber列转换为-180到180
        df['FileNumber'] = ((df['FileNumber'] + 180) % 360) - 180

        # 追加到总的DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# 保存合并后的DataFrame到一个新的CSV文件
combined_df.to_csv('combined_output.csv', index=False)
