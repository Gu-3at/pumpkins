import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

def visualize_data(df):
    # === 5. 可视化分析 ===
    print("\n" + "=" * 40)
    print("数据可视化分析")
    print("=" * 40)

    # 价格趋势图（使用标准化价格）
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Month', y='Std_Price', ci='sd', marker='o', color='darkorange')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('南瓜月度标准化价格趋势(每蒲式耳)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('std_price_trend.png', dpi=300)
    plt.close()

    # 包装与价格关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Bushel_Equivalent', y='Std_Price', hue='Item Size', palette='viridis', alpha=0.7)
    plt.title('包装大小与标准化价格关系')
    plt.savefig('bushel_std_price.png', dpi=300)
    plt.close()

    # 交互式3D图表
    try:
        fig = px.scatter_3d(
            df,
            x='Month',
            y='Origin Group' if 'Origin Group' in df.columns else 'Item Size',
            z='Std_Price',
            color='Item Size' if 'Item Size' in df.columns else 'Season',
            size='Bushel_Equivalent',
            hover_data=['City'] if 'City' in df.columns else None,
            title="南瓜数据多维分析(标准化价格)"
        )
        fig.update_layout(scene=dict(
            xaxis_title='月份',
            yaxis_title='产地' if 'Origin Group' in df.columns else '大小',
            zaxis_title='标准化价格(USD/蒲式耳)'
        ))
        fig.write_html('3d_interactive_std_price.html')
        print("已保存交互式3D图：3d_interactive_std_price.html")

    except Exception as e:
        print(f"交互式图表失败: {e}")

    # 相关性热力图
    print("\n正在生成相关性热力图...")
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 3:
            corr_matrix = df[numeric_cols].corr(numeric_only=True)
            if not corr_matrix.isna().all().all():
                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(
                    corr_matrix,
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt=".2f",
                    linewidths=.5
                )
                plt.title('特征相关性热力图', fontsize=16)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("相关性热力图保存为 correlation_heatmap.png")
            else:
                print("相关性矩阵全为空，跳过可视化")
        else:
            print("数值列不足，无法生成热力图")

    except Exception as e:
        print(f"热力图生成失败: {e}")

    print("\n" + "=" * 40)
    print("分析完成！输出文件:")
    print("静态图表: std_price_trend.png, bushel_std_price.png, correlation_heatmap.png")
    print("交互图表: 3d_interactive_std_price.html")
    print("=" * 40)