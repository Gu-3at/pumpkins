"""
统一调度入口
"""
from data_processing import load_and_preprocess_data
from visualization import visualize_data
from modeling import train_and_evaluate_model
from train_lgbm import train_and_evaluate_lightgbm
from train_XGBoost import train_and_evaluate_xgboost




def main():
    df = load_and_preprocess_data()

    #visualize_data(df)
    train_and_evaluate_model(df)
    #train_and_evaluate_lightgbm(df)
    #train_and_evaluate_xgboost(df)
if __name__ == "__main__":
    main()