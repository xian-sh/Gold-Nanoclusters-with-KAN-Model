import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv(r'D:\GAN1\JTT\GAP-41NEW_swt.csv')

# 选择x和y
x = df.iloc[:, 1:5]  # 第三列至第六列
y = df.iloc[:, 0]    # 第二列

# 将x和y合并为数据框
data_for_heatmap = pd.concat([y, x], axis=1)

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(data_for_heatmap.corr(), annot=True, cmap='coolwarm', annot_kws={"size": 15})
plt.title('Heatmap of Selected Columns', fontsize=15)
plt.savefig('gap.png', dpi=800)
plt.show()