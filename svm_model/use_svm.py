import joblib
import pandas as pd
import numpy as np

def predict_with_model(model_filename, new_data):
    
    loaded_data = joblib.load(model_filename)
    loaded_model = loaded_data['model']
    scaler = loaded_data['scaler']
    print("Model loaded successfully.")

    #print(new_data)
    # 提取模型需要的特征，并将其调整为2D数组
    X_new = np.array([[new_data['LL'], new_data['D_LL'], new_data['Perplexity']]])
    feature_names = ['LL', 'D_LL', 'Perplexity']
    X_new_df = pd.DataFrame(X_new, columns=feature_names)
    # 使用模型进行预测和概率估计
    pred = loaded_model.predict(X_new_df)
    prob = loaded_model.predict_proba(X_new_df)

    print(pred)
    print(prob)

    # 组织结果
    result = {
        'Predicted': pred[0],  # 假设 pred 已经定义
        'AI Probability': prob[0, 1],  # 取第一行的第二个元素
        'Human Probability': prob[0, 0]  # 取第一行的第一个元素
    }

    return result

def predict_with_model_en(model_filename, new_data):
    
    # 加载保存的模型和标准化器
    loaded_data = joblib.load(model_filename)
    loaded_model = loaded_data['model']
    scaler = loaded_data['scaler']
    
    print("Model and scaler loaded successfully.")
    print(new_data)
    # 提取模型需要的特征，并将其调整为2D数组
    X_new = np.array([[new_data['LL'], new_data['Score'], new_data['Perplexity']]])
    feature_names = ['LL', 'Score', 'Perplexity']
    X_new = pd.DataFrame(X_new, columns=feature_names)
    # 标准化新数据
    X_new_scaled = scaler.transform(X_new)
    
    # 使用模型进行预测和概率估计
    pred = loaded_model.predict(X_new_scaled)
    prob = loaded_model.predict_proba(X_new_scaled)
    
    print(pred)
    print(prob)
    
    # 后处理步骤
    human_prob = prob[0, 0]
    ai_prob = prob[0, 1]
    
    if 0.5 <= human_prob < 0.65:
        pred[0] = 1
        human_prob -= 0.15
        ai_prob = 1 - human_prob  # 确保概率和为1
    
    # 组织结果
    result = {
        'Predicted': pred[0],
        'AI Probability': ai_prob,
        'Human Probability': human_prob
    }
    
    return result

def predict_with_model_cn_lines(model_filename, new_data):
    
  # 加载保存的模型
    loaded_model = joblib.load(model_filename)
    print("Model loaded successfully.")
    print("Model loaded successfully.")

    #print(new_data)
    # 提取模型需要的特征，并将其调整为2D数组
    X_new = np.array([[new_data['LL'], new_data['Perplexity']]])
    feature_names = ['LL', 'Perplexity']
    X_new_df = pd.DataFrame(X_new, columns=feature_names)
    # 使用模型进行预测和概率估计
    pred = loaded_model.predict(X_new_df)
    prob = loaded_model.predict_proba(X_new_df)

    print(pred)
    print(prob)

    # 组织结果
    result = {
        'Predicted': pred[0],  # 假设 pred 已经定义
        'AI Probability': prob[0, 1],  # 取第一行的第二个元素
        'Human Probability': prob[0, 0]  # 取第一行的第一个元素
    }

    return result