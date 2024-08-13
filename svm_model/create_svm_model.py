import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import sys

data_filename = sys.argv[1]
output_name = sys.argv[2]
# 读取数据
data= pd.read_csv("matrix/"+data_filename+".txt", header=None)

# 添加列名
columns = ['LL','D_LL', 'Score', 'Perplexity','label','base_model', 'mask_model','length' ,'perturbations','timecost','perturb_timecost','env']

data.columns = columns

# 选择特征和标签
X = data[['LL', 'Perplexity']]
y = data['label']

# 检查是否有无穷大值，并将其替换为 9999
X.replace([np.inf, -np.inf], 9999, inplace=True)

# 检查是否有 NaN 值，并将其替换为 9999
X.fillna(9999, inplace=True)
print("Max values:\n", X.max())
print("Min values:\n", X.min())

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用指定的参数训练SVM模型，启用概率估计
best_params = {'C': 4.1, 'gamma': 0.451}
svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True)
svm.fit(X_train, y_train)

# 导出模型到指定路径
model_filename = "svm_model/svm_model_"+output_name+".joblib"
joblib.dump(svm, model_filename)
print(f"Model saved to {model_filename}")
