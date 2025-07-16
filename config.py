"""
所有模型共享的超参搜索空间与训练配置
"""
RANDOM_STATE = 42
N_JOBS = -1

# LightGBM 搜索空间
LGBM_PARAM_GRID = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [-1, 6, 8],
    'model__learning_rate': [0.05, 0.1],
    'model__num_leaves': [31, 63],
}

# XGBoost 搜索空间
XGB_PARAM_GRID = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [4, 6],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0],
}