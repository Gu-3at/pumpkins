import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold


def train_and_evaluate_model(df):
    # 初始化列表用于存储预测值和真实值
    train_true_values = []
    train_pred_values = []
    test_true_values = []
    test_pred_values = []

    np.random.seed(42)

    # 1. 准备特征和目标变量
    features = [
        'Variety', 'Item Size', 'Repack', 'Year',
        'Month', 'Day', 'Standard_Package', 'Bushel_Equivalent',  # 添加 Day 作为特征
        'City', 'Origin Group'
    ]
    target = 'Avg Price'

    X = df[features]
    y = df[target]

    # 检查重复数据
    print("检查重复数据：")
    print(X.duplicated().sum())

    # 如果有重复数据，可以选择删除
    X = X.drop_duplicates()
    y = y.loc[X.index]  # 确保目标变量与特征对齐

    # 2. 识别特征类型（分类和数值）
    categorical_features = [
        'Variety', 'Item Size', 'Month', 'City', 'Origin Group'
    ]
    numerical_features = ['Year', 'Bushel_Equivalent', 'Day']  # 添加 Day 作为数值特征

    # 3. 检查缺失值
    print("缺失值检查：")
    print(X.isna().sum())

    # 4. 创建预处理管道（处理缺失值）
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

    # 5. 划分数据集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=X[['Year']],
        random_state=42
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 6. 创建带预处理和模型的全管道
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=50,  # 减少树的数量
            max_depth=7,  # 限制深度
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1))
    ])

    # 7. 简化超参数优化（避免网格搜索导致错误）
    params = {
        'model__max_depth': [5, 7],
        'model__min_samples_split': [3, 5],
        'model__max_features': [0.5, 0.7]
    }

    # 使用简化搜索（不使用GridSearchCV避免复杂计算）
    best_score = -np.inf
    best_params = None

    for depth in params['model__max_depth']:
        for split in params['model__min_samples_split']:
            for features_ratio in params['model__max_features']:
                # 设置当前参数组合
                rf_pipeline.set_params(
                    model__max_depth=depth,
                    model__min_samples_split=split,
                    model__max_features=features_ratio
                )

                # 训练模型
                rf_pipeline.fit(X_train, y_train)

                # 评估模型
                score = rf_pipeline.score(X_test, y_test)

                # 检查是否最佳
                if score > best_score:
                    best_score = score
                    best_params = {
                        'max_depth': depth,
                        'min_samples_split': split,
                        'max_features': features_ratio
                    }

    # 8. 使用最佳参数训练最终模型
    print(f"\n最佳参数: {best_params}")
    final_model = rf_pipeline
    final_model.set_params(
        model__max_depth=best_params['max_depth'],
        model__min_samples_split=best_params['min_samples_split'],
        model__max_features=best_params['max_features']
    )
    final_model.fit(X_train, y_train)

    # --- 新增：交叉验证 ---
    print("\n交叉验证：")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(final_model, X, y, cv=kf, scoring='r2')
    print(f"交叉验证 R² 分数：{cv_scores}")
    print(f"平均交叉验证 R² 分数：{cv_scores.mean():.4f}")

    # --- 新增：获取预处理后的特征名称 ---
    # 获取数值特征名称
    numerical_features_names = numerical_features.copy()

    # 获取分类特征的One-Hot编码后的名称
    categorical_encoder = final_model.named_steps['preprocessor'].named_transformers_['cat']
    categorical_features_names = categorical_encoder.named_steps['onehot'].get_feature_names_out(categorical_features)

    # 合并所有特征名称
    all_feature_names = np.concatenate([numerical_features_names, categorical_features_names])

    # 9. 模型评估
    y_pred = final_model.predict(X_test)

    # --- 新增：特征重要性分析 ---
    def analyze_feature_importance(model_pipeline, feature_names):
        # 获取随机森林模型
        rf_model = model_pipeline.named_steps['model']
        # 获取特征重要性
        feature_importances = rf_model.feature_importances_

        # 创建特征重要性 DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('特征重要性排名')
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()

        return importance_df

    importance_df = analyze_feature_importance(final_model, all_feature_names)
    print(importance_df.head(15))

    def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
        # 计算训练集的性能指标
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        train_within_10 = np.mean(np.abs(y_train - y_train_pred) < 0.1 * y_train) * 100

        # 计算测试集的性能指标
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        test_within_10 = np.mean(np.abs(y_test - y_test_pred) < 0.1 * y_test) * 100

        # 打印训练集的性能
        print("===== 训练集模型性能 =====")
        print(f"平均绝对误差 (MAE): {train_mae:.4f}")
        print(f"均方根误差 (RMSE): {train_rmse:.4f}")
        print(f"决定系数 (R²): {train_r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {train_mape:.2f}%")
        print(f"预测误差 < 10%的比例: {train_within_10:.2f}%")

        # 打印测试集的性能
        print("\n===== 测试集模型性能 =====")
        print(f"平均绝对误差 (MAE): {test_mae:.4f}")
        print(f"均方根误差 (RMSE): {test_rmse:.4f}")
        print(f"决定系数 (R²): {test_r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {test_mape:.2f}%")
        print(f"预测误差 < 10%的比例: {test_within_10:.2f}%")

        # 返回训练集和测试集的性能指标
        return {
            'train': {'MAE': train_mae, 'RMSE': train_rmse, 'R2': train_r2, 'MAPE': train_mape,
                      'Within_10pct': train_within_10},
            'test': {'MAE': test_mae, 'RMSE': test_rmse, 'R2': test_r2, 'MAPE': test_mape,
                     'Within_10pct': test_within_10}
        }

    # 生成预测值并收集数据
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)

    # 收集预测值和真实值
    train_true_values = y_train.tolist()
    train_pred_values = y_train_pred.tolist()
    test_true_values = y_test.tolist()
    test_pred_values = y_test_pred.tolist()

    metrics = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)

    # 11. 可视化预测结果 - 线性对比图
    def plot_price_comparison_line(y_true, y_pred, title):
        plt.figure(figsize=(12, 8))
        # 按索引排序以保持原始数据顺序
        sorted_indices = np.argsort(y_true)
        y_true_sorted = np.array(y_true)[sorted_indices]
        y_pred_sorted = np.array(y_pred)[sorted_indices]
        x = np.arange(len(y_true_sorted))

        # 绘制实际价格和预测价格
        plt.plot(x, y_true_sorted, 'b-', linewidth=2.5, label='实际价格', alpha=0.8)
        plt.plot(x, y_pred_sorted, 'r--', linewidth=2, label='预测价格', alpha=0.9)

        # 添加标题和标签
        plt.xlabel('样本编号', fontsize=12)
        plt.ylabel('价格', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # 使用训练集数据绘制对比图
    plot_price_comparison_line(y_train, y_train_pred, '训练集价格对比')

    # 使用测试集数据绘制对比图
    plot_price_comparison_line(y_test, y_pred, '测试集价格对比')

    # 12. 残差分析
    residuals = y_test - y_pred

    # 将 residuals 转换为 Pandas Series，确保索引一致
    residuals = pd.Series(residuals, index=y_test.index)

    # 找出残差绝对值最大的前两个样本
    large_residuals_indices = np.argsort(np.abs(residuals))[-2:]

    # 找出残差绝对值最小的前两个样本
    small_residuals_indices = np.argsort(np.abs(residuals))[:2]

    print("\n残差较大的样本分析：")
    for idx in large_residuals_indices:
        # 获取原始索引
        original_index = X_test.index[idx]

        # 打印信息
        print(f"样本索引: {original_index}")
        print(f"真实值: {y_test.loc[original_index]:.2f}")
        print(f"预测值: {y_pred[idx]:.2f}")
        print(f"残差: {residuals.loc[original_index]:.2f}")
        print(f"特征值: {X_test.loc[original_index]}")
        print("-" * 50)

    print("\n残差较小的样本分析：")
    for idx in small_residuals_indices:
        # 获取原始索引
        original_index = X_test.index[idx]

        # 打印信息
        print(f"样本索引: {original_index}")
        print(f"真实值: {y_test.loc[original_index]:.2f}")
        print(f"预测值: {y_pred[idx]:.2f}")
        print(f"残差: {residuals.loc[original_index]:.2f}")
        print(f"特征值: {X_test.loc[original_index]}")
        print("-" * 50)

    # 13. 条件查询函数 - 返回包含价格的样本
    def find_samples_by_conditions(conditions):
        # 检查所有条件列是否存在
        missing_columns = [col for col in conditions if col not in X.columns]
        if missing_columns:
            print(f"警告: 数据中缺少条件列 {missing_columns}")
            return pd.DataFrame(), pd.DataFrame()

        try:
            # 构建查询条件
            conditions_list = [f"`{col}` == '{val}'" for col, val in conditions.items()]
            query_str = " & ".join(conditions_list)

            # 在训练集中查找符合条件的样本
            train_samples = X_train.query(query_str) if query_str else pd.DataFrame()

            # 在测试集中查找符合条件的样本
            test_samples = X_test.query(query_str) if query_str else pd.DataFrame()

            # 添加真实价格
            if not train_samples.empty:
                train_samples = train_samples.assign(
                    Actual_Price=y_train.loc[train_samples.index].values
                )

            if not test_samples.empty:
                test_samples = test_samples.assign(
                    Actual_Price=y_test.loc[test_samples.index].values
                )

            return train_samples, test_samples

        except Exception as e:
            print(f"查询错误: {e}")
            return pd.DataFrame(), pd.DataFrame()

    # 使用条件查询
    conditions = {
        'City': 'COLUMBIA',
        'Origin Group': 'PENNSYLVANIA',
        'Item Size': 'M',
        'Repack': 'N'
    }

    train_cond_samples, test_cond_samples = find_samples_by_conditions(conditions)

    # 添加预测价格
    if not train_cond_samples.empty:
        # 创建不包含价格列的副本用于预测
        train_cond_for_pred = train_cond_samples.drop(columns=['Actual_Price'], errors='ignore')
        train_cond_samples = train_cond_samples.assign(
            Predicted_Price=final_model.predict(train_cond_for_pred)
        )

    if not test_cond_samples.empty:
        # 创建不包含价格列的副本用于预测
        test_cond_for_pred = test_cond_samples.drop(columns=['Actual_Price'], errors='ignore')
        test_cond_samples = test_cond_samples.assign(
            Predicted_Price=final_model.predict(test_cond_for_pred)
        )

    # 打印结果时显示价格信息
    print("\n训练集中满足条件的样本：")
    if not train_cond_samples.empty:
        # 选择显示的列
        display_cols = ['Actual_Price', 'Predicted_Price', 'Variety',
                        'Standard_Package', 'Bushel_Equivalent']
        # 只保留数据中存在的列
        available_cols = [col for col in display_cols if col in train_cond_samples.columns]
        print(train_cond_samples[available_cols].to_string())
    else:
        print("未找到匹配的训练样本")

    print("\n测试集中满足条件的样本：")
    if not test_cond_samples.empty:
        # 选择显示的列
        display_cols = ['Actual_Price', 'Predicted_Price', 'Variety', 'Year', 'Month', 'Day',
                        'Standard_Package', 'Bushel_Equivalent']
        # 只保留数据中存在的列
        available_cols = [col for col in display_cols if col in test_cond_samples.columns]
        print(test_cond_samples[available_cols].to_string())
    else:
        print("未找到匹配的测试样本")

    # 14. 收集满足条件样本的预测值和真实值
    cond_train_true = []
    cond_train_pred = []
    cond_test_true = []
    cond_test_pred = []

    if not train_cond_samples.empty:
        cond_train_true = train_cond_samples['Actual_Price'].tolist()
        cond_train_pred = train_cond_samples['Predicted_Price'].tolist()

    if not test_cond_samples.empty:
        cond_test_true = test_cond_samples['Actual_Price'].tolist()
        cond_test_pred = test_cond_samples['Predicted_Price'].tolist()

    # 15. 返回所有结果
    return {
        "model": final_model,
        "metrics": metrics,
        "feature_importance": importance_df,
        "train_true": train_true_values,
        "train_pred": train_pred_values,
        "test_true": test_true_values,
        "test_pred": test_pred_values,
        "train_cond_samples": train_cond_samples,
        "test_cond_samples": test_cond_samples,
        "cond_train_true": cond_train_true,
        "cond_train_pred": cond_train_pred,
        "cond_test_true": cond_test_true,
        "cond_test_pred": cond_test_pred,
        "residuals": residuals.values.tolist(),
        "cv_scores": cv_scores.tolist(),
        "best_params": best_params
    }
