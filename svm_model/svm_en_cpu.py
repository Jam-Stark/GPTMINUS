import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def train_and_evaluate_svm_model_cpu(data_path, model_path='svm_model_en_cpu.pkl', test_size=0.3, random_state=123):
    data = pd.read_csv(data_path, sep=",", header=None, names=['D_LL', 'Score', 'Perplexity', 'Perplexity per line', 'label'])

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # 特征和标签
    features = train_data[['D_LL', 'Score', 'Perplexity']]
    labels = train_data['label']
    
    # 训练SVM模型
    svm_model = SVC(kernel='rbf', probability=True, random_state=random_state)
    svm_model.fit(features, labels)
    
    # 保存模型
    joblib.dump(svm_model, model_path)
    
    # 加载模型
    svm_model_loaded = joblib.load(model_path)
    
    # 做预测
    test_features = test_data[['D_LL', 'Score', 'Perplexity']]
    test_labels = test_data['label']
    predictions = svm_model_loaded.predict(test_features)
    
    # 获取预测概率
    probabilities = svm_model_loaded.predict_proba(test_features)
    
    # 输出预测结果和概率
    results = pd.DataFrame({
        'Actual': test_labels,
        'Predicted': predictions,
        'Human_Probability': probabilities[:, 0],
        'AI_Probability': probabilities[:, 1]
    })
    
    print(results)
    
    # 混淆矩阵和准确率
    conf_matrix = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    print(conf_matrix)
    print("Accuracy:", accuracy)

