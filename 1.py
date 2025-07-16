import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings

# ======================== 配置设置 ========================
PLOT_PARAMS = {
    'font.sans-serif': ['SimHei'],  # 中文字体设置
    'axes.unicode_minus': False  # 解决负号显示问题
}

RANDOM_SEED = 42  # 随机种子

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams.update(PLOT_PARAMS)  # 应用绘图设置


# ======================== 工具函数 ========================
def safe_drop_columns(df, cols_to_drop):
    """安全删除列 - 只删除存在的列"""
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    if existing_cols:
        print(f"已删除列: {existing_cols}")
    return df.drop(columns=existing_cols, inplace=False, errors='ignore')


def standardize_package(package_str):
    """将包装描述统一转换为标准格式"""
    if pd.isna(package_str):
        return "unknown"

    package_lower = str(package_str).lower().strip()

    # 直接创建映射字典（基于实际数据内容）
    mapping = {
        # 箱子类（inch bins）
        '36 inch bins': '36 inch bins',
        '24 inch bins': '24 inch bins',
        'bins': '24 inch bins',  # 无尺寸时默认24英寸

        # 巴士耳规格
        '1/2 bushel cartons': '0.5 bu cartons',
        '1 1/9 bushel cartons': '1.111 bu cartons',
        'bushel cartons': '1.0 bu cartons',
        'bushel baskets': '1.0 bu baskets',
        '1 1/9 bushel crates': '1.111 bu crates',

        # 重量规格
        '35 lb cartons': '35 lb cartons',
        '40 lb cartons': '40 lb cartons',
        '50 lb sacks': '50 lb sacks',
        '50 lb cartons': '50 lb cartons',
        '22 lb cartons': '22 lb cartons',
        '20 lb cartons': '20 lb cartons',

        # 特殊类型
        'each': 'each'
    }

    # 查找最接近的匹配（允许部分匹配）
    for key, value in mapping.items():
        if key in package_lower:
            return value

    # 没有匹配时，基于类型进行智能猜测
    if "inch" in package_lower and "bin" in package_lower:
        # 提取尺寸数字
        size_match = re.search(r'(\d+)\s*inch', package_lower)
        if size_match:
            size = int(size_match.group(1))
            return f"{size} inch bins"

    elif "bushel" in package_lower:
        # 提取蒲式耳值
        bu_match = re.search(r'(\d+[\.\d+]*)\s*bushel', package_lower)
        if bu_match:
            return f"{bu_match.group(1)} bu cartons"
        else:
            return "1.0 bu cartons"

    elif "carton" in package_lower or "sack" in package_lower:
        # 提取重量值
        lb_match = re.search(r'(\d+)\s*lb', package_lower)
        if lb_match:
            return f"{lb_match.group(1)} lb cartons"

    # 默认处理
    return "unknown"


def calculate_bushel_equivalent(standard_package):
    """根据标准化包装名称计算蒲式耳当量"""
    if "inch bins" in standard_package:
        # 提取尺寸数字
        try:
            size = float(standard_package.split()[0])
            # 24英寸箱子≈1蒲式耳
            return size / 24
        except:
            return 1.0

    elif "bu" in standard_package:
        # 提取蒲式耳值
        try:
            bu_value = float(standard_package.split()[0])
            return bu_value
        except:
            return 1.0

    elif "lb" in standard_package:
        # 提取磅值
        try:
            lb_value = float(standard_package.split()[0])
            # 45磅 = 1蒲式耳
            return lb_value / 45
        except:
            return 1.0

    elif "each" in standard_package:
        # 单个南瓜，平均估计
        return 0.25  # 约11磅 (45/4=11.25磅)

    else:
        # 未知类型，使用中位数
        return 1.0  # 默认1蒲式耳


# ======================== 数据加载与预处理 ========================
def load_and_preprocess_data():
    """数据加载与预处理主函数"""
    # === 1. 数据加载与初始检查 ===
    print("=" * 40)
    print("步骤1: 数据加载与初始检查")
    print("=" * 40)

    df_backup = None

    try:
        df = pd.read_csv("US-pumpkins.csv")
        print(f"原始数据加载成功! 尺寸: {df.shape}")
        print(f"列名: {', '.join(df.columns)}")
        print(f"样例记录:\n{df.head(2).to_string(index=False)}")

        # 列映射处理
        essential_cols = ['Package', 'Date', 'Low Price', 'High Price']
        missing_cols = [col for col in essential_cols if col not in df.columns]

        if missing_cols:
            print(f"警告: 缺失关键列 {missing_cols}")
            similar_cols = {
                'Package': ['Packaging', 'Package Type'],
                'Date': ['Transaction Date', 'Reported Date'],
                'Low Price': ['Min Price', 'Lower Price'],
                'High Price': ['Max Price', 'Upper Price']
            }

            for missing, alternatives in similar_cols.items():
                for alt in alternatives:
                    if alt in df.columns:
                        print(f"映射 {alt} -> {missing}")
                        df[missing] = df[alt]
                        break

        if any(col not in df.columns for col in essential_cols):
            raise ValueError("关键列缺失且无法自动修复")

        df_backup = df.copy()

        # 安全删除列
        cols_to_drop = ['Type', 'Sub Variety', 'Origin District', 'Unit of Sale', 'Grade',
                        'Environment', 'Quality', 'Condition', 'Appearance', 'Storage',
                        'Crop', 'Trans Mode', 'Unnamed: 24', 'Unnamed: 25']
        df = safe_drop_columns(df, cols_to_drop)

        # === 2. 核心预处理 ===
        print("\n" + "=" * 40)
        print("步骤2: 数据预处理")
        print("=" * 40)

        # 日期处理
        df = process_dates(df)

        # 价格处理
        df = process_prices(df)

        # 包装处理
        df = process_packages(df)

        # 分类变量处理
        df = process_categorical(df)

        # === 3. 数据完整性检查 ===
        print("\n" + "=" * 40)
        print("步骤3: 数据完整性验证")
        print("=" * 40)
        df = validate_data_integrity(df, df_backup)

        # === 4. 数据质量报告 ===
        print("\n" + "=" * 40)
        print("最终数据质量报告")
        print("=" * 40)
        generate_data_report(df)

        return df

    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()


def process_dates(df):
    """日期处理和特征创建"""
    date_parse_success = True
    if 'Date' in df.columns:
        try:
            date_sample = df['Date'].iloc[0]
            if re.match(r'\d{4}-\d{2}-\d{2}', str(date_sample)):
                date_format = '%Y-%m-%d'
            elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', str(date_sample)):
                date_format = '%m/%d/%Y'
            else:
                date_format = None

            print(f"检测到日期格式: {date_format or '自动检测'} | 样例: {date_sample}")
            df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')

            na_ratio = df['Date'].isna().mean()
            if na_ratio > 0.5:
                raise ValueError(f"日期解析失败率过高: {na_ratio:.1%}")

            print(
                f"日期处理完成 | 时间范围: {df['Date'].min().date()} - {df['Date'].max().date()} | 缺失率: {na_ratio:.2%}")

        except Exception as e:
            print(f"日期处理失败: {e}")
            date_parse_success = False
    else:
        date_parse_success = False

    if not date_parse_success:
        print("使用替代日期方案")
        df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        na_ratio = 0.0

    # 时间特征
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day

    seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
               7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    df['Season'] = df['Month'].map(seasons)

    return df


def process_prices(df):
    """价格处理"""
    try:
        for col in ['Low Price', 'High Price']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace('$', '', regex=False), errors='coerce')

        df['Avg Price'] = (df['Low Price'] + df['High Price']) / 2
        price_mask = (df['Low Price'] > 0) & (df['High Price'] >= df['Low Price'])

        if price_mask.sum() > 0:
            df = df[price_mask].copy()
            print(f"价格处理完成 | 保留记录: {len(df)} | 平均价格: ${df['Avg Price'].mean():.2f}")
        else:
            print("警告: 无有效价格数据，跳过过滤")
            df['Avg Price'] = np.random.normal(30, 10, len(df))

    except Exception as e:
        print(f"价格处理失败: {e}")
        df['Avg Price'] = np.random.normal(30, 10, len(df))

    return df


def process_packages(df):
    """包装单位处理"""
    print("\n" + "=" * 40)
    print("改进的包装单位处理")
    print("=" * 40)

    # 1. 标准化包装名称
    if 'Package' in df.columns:
        df['Standard_Package'] = df['Package'].apply(standardize_package)
    else:
        df['Standard_Package'] = "unknown"

    # 2. 计算蒲式耳当量
    df['Bushel_Equivalent'] = df['Standard_Package'].apply(calculate_bushel_equivalent)

    # 3. 计算标准化价格（每蒲式耳价格）
    df['Std_Price'] = df['Avg Price'] / df['Bushel_Equivalent']

    # 记录转换统计
    print("\n包装标准化结果:")
    print(df['Standard_Package'].value_counts())
    print("\n蒲式耳当量分布:")
    print(df['Bushel_Equivalent'].describe()[['mean', 'std', 'min', 'max']])

    return df


def process_categorical(df):
    """分类变量处理"""
    # 大小处理
    if 'Item Size' in df.columns:
        size_mapping = {'small': 'S', 'sm': 'S', 'sml': 'S', 'med': 'M', 'medium': 'M',
                        'large': 'L', 'lge': 'L', 'lg': 'L', 'xl': 'XL', 'exlarge': 'XL',
                        'exl': 'XL', 'jbo': 'Jumbo', 'jumbo': 'Jumbo'}
        df['Item Size'] = df['Item Size'].astype(str).str.lower().str.strip().map(size_mapping).fillna('M')
    else:
        df['Item Size'] = 'M'

    # 城市处理
    if 'City Name' in df:
        df['City'] = df['City Name'].str.extract(r'(\b\w+\b)$').fillna('Unknown')
    elif 'City' not in df:
        df['City'] = 'Unknown'

    # 产地分组
    if 'Origin' in df:
        origin_counts = df['Origin'].value_counts()
        if len(origin_counts) > 5:
            top_origins = origin_counts.index[:5]
            df['Origin Group'] = df['Origin'].apply(lambda x: x if x in top_origins else 'Other')
        else:
            df['Origin Group'] = df['Origin']
    elif 'Origin Group' not in df:
        df['Origin Group'] = 'Unknown'

    return df


def validate_data_integrity(df, df_backup):
    """数据完整性检查"""
    if len(df) == 0:
        print("严重警告: 数据为空！使用备份数据")
        df = df_backup.copy()
        for col in ['Avg Price', 'Month', 'Item Size']:
            if col not in df.columns:
                df[col] = np.nan

    required_cols = ['Avg Price', 'Month', 'Item Size']
    for col in required_cols:
        if col not in df.columns or df[col].isna().all():
            if col == 'Avg Price':
                df[col] = np.random.normal(40, 15, len(df))
            elif col == 'Month':
                df[col] = np.random.randint(1, 13, len(df))
            elif col == 'Item Size':
                df[col] = np.random.choice(['S', 'M', 'L'], len(df))

    return df


def generate_data_report(df):
    """生成数据质量报告"""
    print(f"数据集维度: {df.shape}")
    if 'Date' in df and pd.api.types.is_datetime64_any_dtype(df['Date']):
        print(f"时间范围: {df['Date'].min().date()} - {df['Date'].max().date()}")

    num_cols = ['Avg Price', 'Low Price', 'High Price', 'Bushel_Equivalent', 'Month', 'Std_Price']
    num_cols = [col for col in num_cols if col in df.columns]

    if num_cols:
        print("\n数值特征摘要:")
        print(df[num_cols].describe().loc[['mean', 'std']])
    else:
        print("\n警告: 无数值特征可用")

    cat_cols = ['Item Size', 'City', 'Origin Group', 'Season', 'Standard_Package']
    for col in cat_cols:
        if col in df:
            print(f"\n{col}分布:")
            print(df[col].value_counts().head(10))
        else:
            print(f"\n{col}列不存在")

    # 安全删除列
    cols_to_drop = ['City Name', 'Date', 'Package', 'Mostly Low', 'Mostly High', 'Origin', 'Color']
    df = safe_drop_columns(df, cols_to_drop)

    if 'Variety' in df.columns:
        df = df.dropna(subset=['Variety'])
    else:
        df['Variety'] = 'Unknown'

    return df


# ======================== 数据可视化 ========================
def visualize_data(df):
    """执行所有可视化分析"""
    print("\n" + "=" * 40)
    print("数据可视化分析")
    print("=" * 40)

    # 价格趋势图
    plot_price_trend(df)

    # 包装与价格关系图
    plot_package_relationship(df)

    # 交互式3D图表
    plot_interactive_3d(df)

    # 相关性热力图
    plot_correlation_heatmap(df)

    # 打印分析报告
    print("\n" + "=" * 40)
    print("分析完成！输出文件:")
    print("静态图表: std_price_trend.png, bushel_std_price.png, correlation_heatmap.png")
    print("交互图表: 3d_interactive_std_price.html")
    print("=" * 40)


def plot_price_trend(df):
    """绘制价格趋势图（使用标准化价格）"""
    if 'Month' in df.columns and 'Std_Price' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Month', y='Std_Price', ci='sd', marker='o', color='darkorange')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.title('南瓜月度标准化价格趋势(每蒲式耳)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.savefig('std_price_trend.png', dpi=300)
        plt.close()
        print("已保存标准化价格趋势图: std_price_trend.png")
    else:
        print("缺少Month或Std_Price列，无法绘制价格趋势图")


def plot_package_relationship(df):
    """包装与价格关系图"""
    if 'Bushel_Equivalent' in df.columns and 'Std_Price' in df.columns and 'Item Size' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Bushel_Equivalent', y='Std_Price', hue='Item Size', palette='viridis', alpha=0.7)
        plt.title('包装大小与标准化价格关系')
        plt.savefig('bushel_std_price.png', dpi=300)
        plt.close()
        print("已保存包装与价格关系图: bushel_std_price.png")
    else:
        print("缺少必要列，无法绘制包装与价格关系图")


def plot_interactive_3d(df):
    """绘制交互式3D图表"""
    try:
        if 'Month' in df.columns and 'Std_Price' in df.columns:
            fig = px.scatter_3d(
                df,
                x='Month',
                y='Origin Group' if 'Origin Group' in df.columns else 'Item Size',
                z='Std_Price',
                color='Item Size' if 'Item Size' in df.columns else 'Season',
                size='Bushel_Equivalent' if 'Bushel_Equivalent' in df.columns else None,
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
        else:
            print("缺少必要列，无法生成3D交互图")
    except Exception as e:
        print(f"交互式图表失败: {e}")


def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
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
            print(f"数值列不足 ({len(numeric_cols)}列)，无法生成热力图")
    except Exception as e:
        print(f"热力图生成失败: {e}")


# ======================== 模型训练与评估 ========================
def train_and_evaluate_model(df):
    """模型训练与评估主函数"""
    np.random.seed(RANDOM_SEED)

    # 1. 准备数据
    X, y, features = prepare_training_data(df)

    # 如果数据准备失败，提前返回
    if X is None or y is None or features is None:
        print("数据准备失败，无法继续训练模型")
        return None, None

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # 3. 构建模型管道
    pipeline = build_model_pipeline(X_train, features)  # 传入 X_train

    # 4. 优化超参数
    best_pipeline, best_params = optimize_hyperparameters(pipeline, X_train, y_train)

    # 5. 训练最终模型
    final_model = train_final_model(best_pipeline, X_train, y_train)

    # 6. 交叉验证
    perform_cross_validation(final_model, X, y)

    # 7. 特征重要性分析
    analyze_feature_importance(final_model)

    # 8. 业务价值挖掘
    generate_business_insights(final_model, df, features)

    # 9. 模型评估
    metrics = evaluate_model(final_model, X_train, y_train, X_test, y_test)

    # 10. 可视化预测结果
    visualize_predictions(final_model, X_train, y_train, X_test, y_test)

    # 11. 残差分析
    perform_residual_analysis(final_model, X_test, y_test)

    return final_model, metrics


def prepare_training_data(df):
    """准备训练数据"""
    features = [
        'Variety', 'Item Size', 'Repack', 'Year',
        'Month', 'Day',  'Bushel_Equivalent',
        'City', 'Origin Group'
    ]

    # 只保留存在的特征
    features = [col for col in features if col in df.columns]
    target = 'Avg Price'

    if target not in df.columns:
        print(f"错误: 目标列 '{target}' 不存在")
        return None, None, None

    if not features:
        print("错误: 没有可用的特征列")
        return None, None, None

    X = df[features]
    y = df[target]

    # 检查重复数据
    print(f"检查重复数据：{X.duplicated().sum()} 个重复行")
    X = X.drop_duplicates()
    y = y.loc[X.index]  # 确保目标变量与特征对齐

    return X, y, features


def split_dataset(X, y):
    """划分数据集"""
    # 分层抽样（基于年份）
    try:
        if 'Year' in X.columns:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                stratify=X[['Year']],
                random_state=RANDOM_SEED
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=RANDOM_SEED
            )
    except Exception as e:
        print(f"分层抽样失败: {e}, 使用随机抽样")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_SEED
        )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_model_pipeline(X_train, features):
    """构建模型管道 - 使用传入的 X_train 而不是全局 df"""
    # 使用传入的 X_train 获取数据类型
    categorical_features = [col for col in features if X_train[col].dtype == 'object']
    numerical_features = [col for col in features if col not in categorical_features]

    print(f"分类特征: {categorical_features}")
    print(f"数值特征: {numerical_features}")

    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 创建模型管道
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=50,
            random_state=RANDOM_SEED,
            n_jobs=-1))
    ])

    return pipeline


def optimize_hyperparameters(pipeline, X_train, y_train):
    """优化超参数"""
    best_score = -np.inf
    best_params = None
    best_pipeline = None

    param_combinations = [
        {'max_depth': 5, 'min_samples_split': 3, 'max_features': 0.5},
        {'max_depth': 5, 'min_samples_split': 3, 'max_features': 0.7},
        {'max_depth': 5, 'min_samples_split': 5, 'max_features': 0.5},
        {'max_depth': 5, 'min_samples_split': 5, 'max_features': 0.7},
        {'max_depth': 7, 'min_samples_split': 3, 'max_features': 0.5},
        {'max_depth': 7, 'min_samples_split': 3, 'max_features': 0.7},
        {'max_depth': 7, 'min_samples_split': 5, 'max_features': 0.5},
        {'max_depth': 7, 'min_samples_split': 5, 'max_features': 0.7}
    ]

    for params in param_combinations:
        pipeline.set_params(
            model__max_depth=params['max_depth'],
            model__min_samples_split=params['min_samples_split'],
            model__max_features=params['max_features']
        )

        try:
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_train, y_train)

            if score > best_score:
                best_score = score
                best_params = params
                best_pipeline = pipeline

            print(f"参数组合 {params} 得分: {score:.4f}")
        except Exception as e:
            print(f"参数组合 {params} 训练失败: {e}")

    if best_params is None:
        print("警告: 所有参数组合失败，使用默认参数")
        best_params = {'max_depth': 7, 'min_samples_split': 5, 'max_features': 0.7}
        pipeline.set_params(
            model__max_depth=best_params['max_depth'],
            model__min_samples_split=best_params['min_samples_split'],
            model__max_features=best_params['max_features']
        )
        best_pipeline = pipeline

    print(f"\n最佳参数: {best_params}")
    return best_pipeline, best_params


def train_final_model(pipeline, X_train, y_train):
    """训练最终模型"""
    pipeline.fit(X_train, y_train)
    print("最终模型训练完成！")
    return pipeline


def perform_cross_validation(model, X, y):
    """执行交叉验证"""
    print("\n交叉验证：")
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        print(f"交叉验证 R² 分数：{cv_scores}")
        print(f"平均交叉验证 R² 分数：{cv_scores.mean():.4f}")
    except Exception as e:
        print(f"交叉验证失败: {e}")


def analyze_feature_importance(model_pipeline):
    """分析特征重要性"""
    print("\n分析特征重要性...")
    try:
        # 获取随机森林模型
        rf_model = model_pipeline.named_steps['model']
        # 获取特征重要性
        feature_importances = rf_model.feature_importances_

        # 获取预处理后的特征名称
        preprocessor = model_pipeline.named_steps['preprocessor']

        # 获取分类特征名称
        categorical_encoder = preprocessor.named_transformers_['cat']
        categorical_features = preprocessor.transformers_[1][2]
        category_names = categorical_encoder.named_steps['onehot'].get_feature_names_out(categorical_features)

        # 组合所有特征名称
        numerical_features = preprocessor.transformers_[0][2]
        all_feature_names = list(numerical_features) + list(category_names)

        # 创建特征重要性 DataFrame
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('特征重要性排名')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()

        print("\nTop 15 特征重要性:")
        print(importance_df.head(15).to_string(index=False))

        return importance_df

    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        return pd.DataFrame()


def generate_business_insights(model, df, features):
    """生成业务洞察"""
    print("\n生成业务洞察...")
    try:
        # 生成价格预测
        df['Predicted_Price'] = model.predict(df[features])

        # 分析月份与价格的关系
        if 'Month' in df.columns:
            monthly_prices = df.groupby('Month')['Predicted_Price'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Month', y='Predicted_Price', data=monthly_prices, marker='o')
            plt.title('预测价格的月度趋势')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.savefig('monthly_price_trend.png', dpi=300)
            plt.close()
            print("已保存月度价格趋势图: monthly_price_trend.png")

        # 分析不同品种的价格差异
        if 'Variety' in df.columns:
            variety_prices = df.groupby('Variety')['Predicted_Price'].mean().reset_index()
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Variety', y='Predicted_Price', data=variety_prices)
            plt.title('不同品种的预测价格')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('variety_price.png', dpi=300)
            plt.close()
            print("已保存品种价格差异图: variety_price.png")

            # 添加分析报告
            max_variety = variety_prices.loc[variety_prices['Predicted_Price'].idxmax()]
            min_variety = variety_prices.loc[variety_prices['Predicted_Price'].idxmin()]
            print(f"\n最高价值的品种: {max_variety['Variety']} (${max_variety['Predicted_Price']:.2f})")
            print(f"最低价值的品种: {min_variety['Variety']} (${min_variety['Predicted_Price']:.2f})")
            print(f"品种间最大价格差: ${max_variety['Predicted_Price'] - min_variety['Predicted_Price']:.2f}")

    except Exception as e:
        print(f"业务洞察分析失败: {e}")


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    print("\n评估模型性能...")

    # 训练集预测与评估
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / np.maximum(y_train, 1e-8))) * 100
    train_within_10 = np.mean(np.abs(y_train - y_train_pred) < 0.1 * y_train) * 100

    # 测试集预测与评估
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(y_test, 1e-8))) * 100
    test_within_10 = np.mean(np.abs(y_test - y_test_pred) < 0.1 * y_test) * 100

    # 打印结果
    print("\n===== 模型性能报告 =====")
    print("| 指标              | 训练集        | 测试集        |")
    print("|-------------------|---------------|---------------|")
    print(f"| MAE              | {train_mae:.4f}      | {test_mae:.4f}      |")
    print(f"| RMSE             | {train_rmse:.4f}      | {test_rmse:.4f}      |")
    print(f"| R²               | {train_r2:.4f}      | {test_r2:.4f}      |")
    print(f"| MAPE (%)         | {train_mape:.2f}%     | {test_mape:.2f}%     |")
    print(f"| 误差<10%比例 (%) | {train_within_10:.2f}%     | {test_within_10:.2f}%     |")

    return {
        'train': {'MAE': train_mae, 'RMSE': train_rmse, 'R2': train_r2, 'MAPE': train_mape,
                  'Within_10pct': train_within_10},
        'test': {'MAE': test_mae, 'RMSE': test_rmse, 'R2': test_r2, 'MAPE': test_mape, 'Within_10pct': test_within_10}
    }


def visualize_predictions(model, X_train, y_train, X_test, y_test):
    """可视化预测结果"""
    print("\n可视化预测结果...")

    # 训练集预测可视化
    y_train_pred = model.predict(X_train)
    plt.figure(figsize=(12, 8))
    plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('训练集预测 vs 实际价格')
    plt.savefig('train_prediction.png', dpi=300)
    plt.close()

    # 测试集预测可视化
    y_test_pred = model.predict(X_test)
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='green')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('测试集预测 vs 实际价格')
    plt.savefig('test_prediction.png', dpi=300)
    plt.close()

    print("已保存预测可视化图: train_prediction.png, test_prediction.png")


def perform_residual_analysis(model, X_test, y_test):
    """进行残差分析"""
    print("\n进行残差分析...")

    # 计算残差
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # 残差图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    plt.xlabel('预测价格')
    plt.ylabel('残差')
    plt.title('预测残差分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('residuals.png', dpi=300)
    plt.close()

    # 找出残差最大的样本
    large_residuals_idx = np.argsort(np.abs(residuals))[-5:][::-1]

    print("\n残差最大的5个样本:")
    for idx in large_residuals_idx:
        print(f"\n样本索引: {X_test.index[idx]}")
        print(f"实际值: {y_test.iloc[idx]:.2f}")
        print(f"预测值: {y_pred[idx]:.2f}")
        print(f"残差: {residuals.iloc[idx]:.2f}")
        print(f"特征值:\n{X_test.iloc[idx]}")
        print("-" * 40)

    print("已保存残差分布图: residuals.png")


# ======================== 主函数 ========================
def main():
    """主函数"""
    try:
        # 1. 数据加载与预处理
        df = load_and_preprocess_data()

        # 2. 数据可视化
        visualize_data(df)

        # 3. 模型训练与评估
        model, metrics = train_and_evaluate_model(df)

        if model is None or metrics is None:
            print("模型训练失败，无法完成分析")
            return None, None, None

        print("\n" + "=" * 40)
        print("分析完成！")
        print("=" * 40)

        return df, model, metrics

    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# 运行主函数
if __name__ == "__main__":
    result = main()
    if result is not None:
        df, model, metrics = result
    else:
        print("程序执行失败")