import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime


def load_and_preprocess_data():
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # === 1. 数据加载与初始检查 ===
    print("=" * 40)
    print("步骤1: 数据加载与初始检查")
    print("=" * 40)

    try:
        df = pd.read_csv("US-pumpkins.csv")
        print(f"原始数据加载成功! 尺寸: {df.shape}")
        print(f"列名: {', '.join(df.columns)}")
        print(f"样例记录:\n{df.head(2).to_string(index=False)}")

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

    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

    df_backup = df.copy()

    print(df.info())
    # 删除无效列
    df.drop(columns=['Type', 'Sub Variety', 'Origin District', 'Unit of Sale', 'Grade', 'Environment', 'Quality',
                     'Condition', 'Appearance',
                     'Storage', 'Crop', 'Trans Mode', 'Unnamed: 24', 'Unnamed: 25'], inplace=True)
    print(df.info())

    # === 2. 核心预处理 ===
    print("\n" + "=" * 40)
    print("步骤2: 数据预处理")
    print("=" * 40)

    # 日期处理
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
        df['Date'] = df_backup['Date']
        df['Day'] = np.arange(len(df))
        df['Month'] = np.random.randint(1, 13, len(df))

    # 时间特征
    df['Month'] = df['Date'].dt.month if 'Date' in df and df['Date'].dtype == 'datetime64[ns]' else df['Month']
    df['Year'] = df['Date'].dt.year if 'Date' in df and df['Date'].dtype == 'datetime64[ns]' else datetime.now().year
    df['Day'] = df['Date'].dt.day if 'Date' in df and df['Date'].dtype == 'datetime64[ns]' else np.random.randint(1, 32,
                                                                                                                  len(df))  # 添加日期天数

    seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
               7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    df['Season'] = df['Month'].map(seasons)

    # 价格处理
    try:
        for col in ['Low Price', 'High Price']:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Avg Price'] = (df['Low Price'] + df['High Price']) / 2
        price_mask = (df['Low Price'] > 0) & (df['High Price'] >= df['Low Price'])

        if price_mask.sum() > 0:
            df = df[price_mask]
            print(f"价格处理完成 | 保留记录: {len(df)} | 平均价格: ${df['Avg Price'].mean():.2f}")
        else:
            print("警告: 无有效价格数据，跳过过滤")

    except Exception as e:
        print(f"价格处理失败: {e}")
        df['Avg Price'] = np.random.normal(30, 10, len(df))

    # === 改进的包装单位处理 ===
    print("\n" + "=" * 40)
    print("改进的包装单位处理")
    print("=" * 40)

    def standardize_package(package_str):
        """将各种包装描述统一转换为标准格式"""
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
            size = float(standard_package.split()[0])
            # 24英寸箱子≈1蒲式耳
            return size / 24

        elif "bu" in standard_package:
            # 提取蒲式耳值
            bu_value = float(standard_package.split()[0])
            return bu_value

        elif "lb" in standard_package:
            # 提取磅值
            lb_value = float(standard_package.split()[0])
            # 45磅 = 1蒲式耳
            return lb_value / 45

        elif "each" in standard_package:
            # 单个南瓜，平均估计
            return 0.25  # 约11磅 (45/4=11.25磅)

        else:
            # 未知类型，使用中位数
            return 1.0  # 默认1蒲式耳

    # 应用包装标准化
    try:
        # 1. 标准化包装名称
        df['Standard_Package'] = df['Package'].apply(standardize_package)

        # 2. 计算蒲式耳当量
        df['Bushel_Equivalent'] = df['Standard_Package'].apply(calculate_bushel_equivalent)

        # 3. 计算标准化价格（每蒲式耳价格）
        df['Std_Price'] = df['Avg Price'] / df['Bushel_Equivalent']

        # 记录转换统计
        print("\n包装标准化结果:")
        print(df['Standard_Package'].value_counts())
        print("\n蒲式耳当量分布:")
        print(df['Bushel_Equivalent'].describe()[['mean', 'std', 'min', 'max']])

    except Exception as e:
        print(f"包装处理失败: {e}")
        df['Standard_Package'] = "unknown"
        df['Bushel_Equivalent'] = 1.0
        df['Std_Price'] = df['Avg Price']

    # 分类变量处理
    size_mapping = {'small': 'S', 'sm': 'S', 'sml': 'S', 'med': 'M', 'medium': 'M',
                    'large': 'L', 'lge': 'L', 'lg': 'L', 'xl': 'XL', 'exlarge': 'XL', 'exl': 'XL', 'jbo': 'Jumbo',
                    'jumbo': 'Jumbo'}
    df['Item Size'] = df['Item Size'].astype(str).str.lower().str.strip().map(size_mapping).fillna('M')

    if 'City Name' in df:
        df['City'] = df['City Name'].str.extract(r'(\b\w+\b)$')

    if 'Origin' in df:
        origin_counts = df['Origin'].value_counts()
        if len(origin_counts) > 5:
            top_origins = origin_counts.index[:5]
            df['Origin Group'] = df['Origin'].apply(lambda x: x if x in top_origins else 'Other')
        else:
            df['Origin Group'] = df['Origin']

    # === 3. 数据完整性检查 ===
    print("\n" + "=" * 40)
    print("步骤3: 数据完整性验证")
    print("=" * 40)

    if len(df) == 0:
        print("严重警告: 数据为空！使用备份数据")
        df = df_backup.copy()
        for col in ['Avg Price', 'Month', 'Item Size']:
            if col not in df.columns:
                df[col] = np.nan

    required_cols = ['Avg Price', 'Month', 'Item Size']
    for col in required_cols:
        if df[col].isna().all():
            if col == 'Avg Price':
                df[col] = np.random.normal(40, 15, len(df))
            elif col == 'Month':
                df[col] = np.random.randint(1, 13, len(df))
            elif col == 'Item Size':
                df[col] = np.random.choice(['S', 'M', 'L'], len(df))

    # === 4. 数据质量报告 ===
    print("\n" + "=" * 40)
    print("最终数据质量报告")
    print("=" * 40)

    print(f"数据集维度: {df.shape}")
    print(f"时间范围: {df.get('Date', 'N/A')}")

    num_cols = ['Avg Price', 'Low Price', 'High Price', 'Bushel_Equivalent', 'Month', 'Std_Price']
    num_summary = df[df.columns.intersection(num_cols)].describe().loc[['mean', 'std']]
    if not num_summary.empty:
        print("\n数值特征摘要:")
        print(num_summary)

    cat_cols = ['Item Size', 'City', 'Origin Group', 'Season', 'Standard_Package']
    for col in cat_cols:
        if col in df:
            print(f"\n{col}分布:")
            print(df[col].value_counts().head(10))

    print(df.info())
    df.drop(columns=['City Name', 'Date', 'Package', 'Mostly Low', 'Mostly High', 'Origin', 'Color'], inplace=True)
    df = df.dropna(subset=['Variety'])

    return df