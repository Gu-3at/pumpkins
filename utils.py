import graphviz
from sklearn import tree

def visualize_decision_tree(pipeline, preprocessor, feature_names):
    """使用Graphviz可视化随机森林中的一棵决策树"""
    print("\n正在生成决策树可视化...")

    # 从管道中获取预处理后的特征名称
    try:
        # 获取one-hot编码器
        onehot_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

        # 获取类别特征名称
        categorical_features = preprocessor.transformers_[1][2]  # 分类特征的列名
        category_names = onehot_encoder.get_feature_names_out(input_features=categorical_features)

        # 组合所有特征名称
        numerical_features = preprocessor.transformers_[0][2]  # 数值特征的列名
        all_feature_names = list(numerical_features) + list(category_names)
    except Exception as e:
        print(f"无法获取特征名称: {e}")
        all_feature_names = [f"feature_{i}" for i in range(len(feature_names))]

    # 从随机森林中提取第一棵树
    rf_model = pipeline.named_steps['model']
    estimator = rf_model.estimators_[0]

    # 创建决策树可视化（修正了class_names参数）
    dot_data = tree.export_graphviz(
        estimator,
        out_file=None,
        feature_names=all_feature_names,
        filled=True,  # 使用颜色填充显示类别
        rounded=True,  # 圆角节点
        special_characters=True,  # 处理特殊字符
        proportion=True,  # 显示比例而非计数
    )

    # 生成并保存图像
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render('decision_tree', view=False)  # 保存为文件而不立即打开
    print("已保存决策树可视化: decision_tree.png")
