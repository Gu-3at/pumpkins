import pandas as pd
import numpy as np

# 读取数据（根据实际路径修改）
df = pd.read_csv("US-pumpkins.csv")

# 关键字段筛选与处理
df = df[['Package', 'Date', 'Low Price', 'High Price', 'City Name', 'Origin', 'Variety', 'Item Size']]
df['Date'] = pd.to_datetime(df['Date'])  # 转换日期格式
df['Month'] = df['Date'].dt.month  # 提取月份
df['Avg Price'] = (df['Low Price'] + df['High Price']) / 2  # 计算平均价格

# 过滤以"bushel"为单位的记录
df = df[df['Package'].str.contains('bushel', case=False, na=False)]

# 包装单位标准化
df['Bushel Type'] = df['Package'].str.extract(r'(\d+\s?\d*/\d+)')[0]

# 缺失值处理
df['Item Size'].fillna('med', inplace=True)  # 填充尺寸缺失值
print("数据清洗后维度:", df.shape)

print(df.describe())  # 数值型字段统计
print(df['Bushel Type'].value_counts())  # 包装类型分布

import matplotlib.pyplot as plt
import seaborn as sns

# 每月平均价格折线图
plt.figure(figsize=(12, 6))
monthly_avg = df.groupby('Month')['Avg Price'].mean()
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', color='darkorange')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Monthly Average Pumpkin Price (USD/bushel)', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# 包装类型与价格关系（箱线图）
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Bushel Type', y='Avg Price', palette='viridis')
plt.title('Price Distribution by Bushel Type')
plt.show()

# 产地分布（计数图）
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='Origin', order=df['Origin'].value_counts().index, palette='Blues_r')
plt.title('Pumpkin Origin Distribution')
plt.show()

# 品种与价格关系（分面散点图）
g = sns.FacetGrid(df, col='Variety', col_wrap=3, height=4, sharey=False)
g.map(sns.scatterplot, 'Month', 'Avg Price', alpha=0.7)
g.set_titles("{col_name}")
plt.show()


import plotly.express as px

# 动态价格-时间趋势图
fig = px.scatter(
    df, x='Date', y='Avg Price',
    color='City Name', hover_data=['Origin', 'Variety'],
    title="南瓜价格时空分布",
    trendline='lowess'  # 添加局部回归趋势线
)
fig.show()

# 3D产地-品种-价格关系
fig = px.scatter_3d(
    df, x='Origin', y='Variety', z='Avg Price',
    color='Item Size', symbol='Bushel Type',
    title="多维特征关联分析"
)
fig.update_layout(scene=dict(zaxis=dict(title='价格（美元）')))
fig.show()


# 分类变量编码
df_encoded = pd.get_dummies(df, columns=['City Name', 'Origin', 'Item Size', 'Variety'])

# 时间特征衍生
df_encoded['Day_of_Year'] = df['Date'].dt.dayofyear  # 年序日期
df_encoded['Is_Harvest_Season'] = df['Month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)  # 收获季标志

# 数据预处理（确保所有特征为数值型）
import pandas as pd

# 1. 转换分类特征为数值编码
size_mapping = {'bushel': 1, 'cartons': 2}  # 示例映射
df_encoded['Package_Type'] = df_encoded['Package'].map(size_mapping)

# 步骤1：检查列名
print("当前列名:", df_encoded.columns.tolist())

# 步骤2：安全删除（推荐）
df_numeric = df_encoded.drop(columns=['Package', 'Description'], errors='ignore')

# 步骤3：后续操作（例如计算相关系数）
corr_matrix = df_numeric.corr(numeric_only=True)  # 确保仅处理数值列


# 4. 可视化热力图
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 尝试不同字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 3. 创建美观的热力图
plt.figure(figsize=(14, 12), dpi=100)  # 增大画布尺寸

# 创建mask隐藏上三角（避免重复信息）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 绘制热力图
ax = sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',  # 更鲜明的双色渐变
    center=0,  # 以0为中心值
    annot=True,  # 显示数值
    fmt=".2f",  # 保留2位小数
    annot_kws={'size': 10},  # 注释字体大小
    linewidths=.5,  # 单元格间线宽
    cbar_kws={'label': '相关系数'}  # 添加色标注解
)

# 4. 优化标签展示
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
    fontsize=12
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=12
)

# 5. 添加标题
plt.title('南瓜数据集特征相关性热力图', fontsize=16, pad=20)

# 6. 解决标签截断问题
plt.tight_layout()

# 保存高质量图片
plt.savefig('optimized_correlation_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()

