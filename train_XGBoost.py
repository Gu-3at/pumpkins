import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def train_and_evaluate_xgboost(df):
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
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 7. 简化超参数优化（避免网格搜索导致错误）
    params = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8, 1.0]
    }

    # 使用简化搜索（不使用GridSearchCV避免复杂计算）
    best_score = -np.inf
    best_params = None

    for ne in params['model__n_estimators']:
        for md in params['model__max_depth']:
            for lr in params['model__learning_rate']:
                for ss in params['model__subsample']:
                    xgb_pipeline.set_params(
                        model__n_estimators=ne,
                        model__max_depth=md,
                        model__learning_rate=lr,
                        model__subsample=ss
                    )

                    # 训练模型
                    xgb_pipeline.fit(X_train, y_train)

                    # 评估模型
                    score = xgb_pipeline.score(X_test, y_test)

                    # 检查是否最佳
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_estimators': ne,
                            'max_depth': md,
                            'learning_rate': lr,
                            'subsample': ss
                        }

    # 8. 使用最佳参数训练最终模型
    print(f"\n最佳参数: {best_params}")
    final_model = xgb_pipeline
    final_model.set_params(
        model__n_estimators=best_params['n_estimators'],
        model__max_depth=best_params['max_depth'],
        model__learning_rate=best_params['learning_rate'],
        model__subsample=best_params['subsample']
    )
    final_model.fit(X_train, y_train)

    # --- 新增：交叉验证 ---
    print("\n交叉验证：")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(final_model, X, y, cv=kf, scoring='r2')
    print(f"交叉验证 R² 分数：{cv_scores}")
    print(f"平均交叉验证 R² 分数：{cv_scores.mean():.4f}")

    # 9. 模型评估
    y_pred = final_model.predict(X_test)

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

    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    metrics = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)

    return final_model, metrics