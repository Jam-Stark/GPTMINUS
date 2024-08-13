import time
import sys
import numpy as np
import joblib
import pandas as pd
import unicodedata
import torch

from svm_model.use_svm import predict_with_model,predict_with_model_en,predict_with_model_cn_lines
from Mytools.check_2long import split_texts

def write_values_to_file(filename, values):
    # 在最后添加内容，不抹去原始内容
    with open(filename, 'a') as file:  # 使用追加模式打开文件
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        #file.write(str(values) + "\n")  # 写入内容并换行
        for x in values:
            file.write(str(x) + "\n")

def read_texts_from_file(filename):
    type=False
    with open(filename, 'r',encoding='utf-8') as file:
        content = file.read()
    if is_chinese(content):
        texts=split_texts(content)
        type=True
        texts = content.split('</text>')
        texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
        #print(texts)
    else:
        texts = content.split('</text>')
        texts = [text.replace('<text>', '') for text in texts if text.strip() != '']

    return texts,type

def process_texts(content):
    type=False
    if is_chinese(content):
        texts=split_texts(content)
        type=True
        texts = content.split('</text>')
        texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
        #print(texts)
    else:
        texts = content.split('</text>')
        texts = [text.replace('<text>', '') for text in texts if text.strip() != '']

    return texts,type

def extract_to_matrix(data, keys_order):
    data = process_mapping(data)
    # Extract values based on the specified order of keys
    values = [data[key] for key in keys_order]
    
    # Convert the list of values to a numpy array (1 row, n columns matrix)
    matrix = np.array([values])
    
    return matrix
    
def append_matrix_to_file(matrix, filename):
    # Convert the matrix to a flattened string and remove brackets
    matrix_str = ', '.join(map(str, matrix.flatten()))
    
    # Prepare the content to write
    content = f"{matrix_str}\n"
    
    # Open the file in append mode and write the content
    with open(filename, 'a') as file:
        file.write(content)

def is_chinese(text):
    #print(text)
    english_count = 0
    chinese_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
        elif unicodedata.category(char).startswith('L'):
            english_count += 1
    
    if chinese_count > english_count:
        return True
    elif english_count > chinese_count:
        return False

def process_mapping(results):
    """ 模型映射表
    01 gpt2-large   05 t5-large  09 wenzhong
    02 gpt2-medium  06 t5-small  10 qwen
    03 gpt2-xl      07 t5-3b     11 blueLM
    04 gpt2-small   08 deepseek  12 bert
    """
    mapping = {
        'gpt2-large': '01',
        'gpt2-medium': '02',
        'gpt2-xl': '03',
        'gpt2-small': '04',
        't5-large': '05',
        't5-small': '06',
        't5-3b': '07',
        'wenzhong': '08',
        'qwen': '09',
        'blueLM': '11',
        'bert': '12',
        'deepseek': '08',

        'none': '20',
        'paper': '21',
        'science&tech': '22',
        'life&news': '23',
        'literature': '24',
        'history': '25',
        'laws': '26',
        'economy': '27',
    }
    
    print(results)
    results['base_model'] = mapping[results['base_model']]
    results['mask_model'] = mapping[results['mask_model']]

    results['env'] = mapping[results['env']]

    return results

def predict_model(content,_mask_model_id,if_analysis_sentence,env):

    from test_en_gpu import PPL_LL_based_gpt2_t5
    from test_cn import PPL_LL_based_cn
    from svm_model.use_svm import predict_with_model
    values=[]
    texts,language=process_texts(content)
    print(len(texts))

    if(language):
        model_cn=PPL_LL_based_cn(model_id="qwen")
        predictions=[]
        for i in range(len(texts)):
            print("cn")
            print("第",i+1,"次预测")
            print(texts[i])
            results=model_cn(texts[i],mask_model_id=_mask_model_id,if_analysis_sentences=int(if_analysis_sentence),n_perturbations=15)
            if int(if_analysis_sentence)==0:
                if _mask_model_id=="deepseek":
                    prediction=predict_with_model("svm_model/svm_model_cn_deepseek.joblib",results)
                elif _mask_model_id=="blueLM":
                    prediction=predict_with_model("svm_model/svm_model_cn_blueLM.joblib",results)
                predictions.append(prediction)
                results['label']=prediction['Predicted']
                results['env']=env

                keys_order_cn = ['LL','D_LL', 'Score', 'Perplexity','Perplexity per line','label','base_model', 'mask_model','length' ,'perturbations','timecost','perturb_timecost','env']
                m=extract_to_matrix(results,keys_order_cn)
                
                append_matrix_to_file(m,"matrix/cache_matrix_cn.txt")
                values.append(results)
                write_values_to_file('results.txt', values)
                values.clear()
            else:
                for result in results:
                    if _mask_model_id=="deepseek":
                        prediction=predict_with_model_cn_lines("svm_model/svm_model_cn_deepseek_lines.joblib",result)
                        print(result['sentence'])
                        prediction['sentence']=result['sentence']
                    elif _mask_model_id=="blueLM":
                        prediction=predict_with_model_cn_lines("svm_model/svm_model_cn_blueLM_lines.joblib",result)
                        prediction['sentence']=result['sentence']
                    predictions.append(prediction)
                    result['label']=prediction['Predicted']
                    result['env']=env

                    keys_order_cn = ['LL','D_LL', 'Score', 'Perplexity','label','base_model', 'mask_model','length' ,'perturbations','timecost','perturb_timecost','env']
                    m=extract_to_matrix(result,keys_order_cn)
                    append_matrix_to_file(m,"matrix/cache_matrix_cn_lines.txt")
                    values.append(result)
                    write_values_to_file('results.txt', values)
                    values.clear()
        print(predictions)
        return predictions

    else:
        model_en = PPL_LL_based_gpt2_t5()
        predictions=[]
        for i in range(len(texts)):
            print("en")
            print("第", i + 1, "次预测")
            print(texts[i])
            if len(texts[i])>1024:
                texts[i]=texts[i][:1024]
            results = model_en(texts[i],model_id="gpt2-large",mask_model_id=_mask_model_id,n_perturbations=30)
            #print(type(results))
            prediction=predict_with_model_en("svm_model/svm_model_en.joblib",results)
            predictions.append(prediction)
            results['label']=prediction['Predicted']
            results['env']=env

            keys_order_en = ['LL','D_LL', 'Score','Perplexity','Perplexity per line', 'label','base_model', 'mask_model', 'length','perturbations','env']
            m=extract_to_matrix(results,keys_order_en)
            append_matrix_to_file(m,"matrix/cache_matrix_en.txt")
            values.append(results)
            write_values_to_file('results.txt', values)
            values.clear()
    return predictions
