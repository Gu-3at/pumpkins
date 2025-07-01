import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
pumpkins = pd.read_csv('US-pumpkins.csv')

# 提取关键列并计算月份
pumpkins = pumpkins[['Package', 'Date', 'Low Price', 'High Price']]
pumpkins['Date'] = pd.to_datetime(pumpkins['Date'])
pumpkins['Month'] = pumpkins['Date'].dt.month

# 计算平均价格
pumpkins['Avg Price'] = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

# 仅保留以"bushel"为单位的记录
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=False)]

# 计算每月平均价格
monthly_avg = pumpkins.groupby('Month')['Avg Price'].mean().reset_index()

#每月平均价格趋势（折线图）
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg, x='Month', y='Avg Price', marker='o', color='darkorange')
plt.title('Monthly Average Pumpkin Price (USD/bushel)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Average Price (USD)')
plt.xticks(range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(alpha=0.3)
plt.show()

#价格分布箱线图（按月份）
plt.figure(figsize=(12, 6))
sns.boxplot(data=pumpkins, x='Month', y='Avg Price', palette='autumn')
plt.title('Distribution of Pumpkin Prices by Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Price (USD)')
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

#包装类型与价格关系（条形图)

# 提取包装类型
pumpkins['Bushel Type'] = pumpkins['Package'].str.extract(r'(\d+\s?\d*/\d+)')[0]  # 匹配如"1 1/9"或"1/2"

plt.figure(figsize=(10, 6))
sns.barplot(data=pumpkins, x='Bushel Type', y='Avg Price', ci=None, palette='viridis')
plt.title('Average Price by Bushel Package Type', fontsize=14)
plt.xlabel('Bushel Measurement')
plt.ylabel('Average Price (USD)')
plt.show()

#价格-月份散点图 + 回归趋势线
plt.figure(figsize=(12, 6))
sns.regplot(data=pumpkins, x='Month', y='Avg Price', scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
plt.title('Price vs. Month with Regression Line', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Price (USD)')
plt.xticks(range(1, 13))
plt.grid(alpha=0.2)
plt.show()

#多维分面分析（包装类型×月份）
g = sns.FacetGrid(pumpkins, col='Bushel Type', col_wrap=3, height=4, aspect=1.2)
g.map_dataframe(sns.boxplot, x='Month', y='Avg Price', palette='Set2')
g.set_axis_labels('Month', 'Average Price (USD)')
g.set_titles('{col_name} Bushel')
g.fig.suptitle('Price Distribution by Month and Bushel Type', y=1.05, fontsize=16)
plt.xticks(rotation=45)
plt.show()