import base64
import streamlit as st
import time
import cv2
import pytesseract  # OCR:图片转文字
import numpy as np
import speech_recognition as sr  # 语音转文字
from run_new import predict_model
import re

import requests
import base64
import uuid
import time

from Mytools.auth_util import gen_sign_headers

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

def set_page_style():
    st.markdown(""" 
    <style>
    body {
        background-color: #BFD4D3;
        font-family: Arial, sans-serif;
    }
    .pink-box {
        background-color: #F0C9CF;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .sentence-box {
        background-color: #E6D2D5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .highlighted-sentence {
        color: black;
        background-color: #FFFF00;
        padding: 2px;
        border-radius: 3px;
    }
    .bold-red {
        font-weight: bold;
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)

def show_progress(text='正在检测中', current_step=0, total_steps=100):
    progress_bar = st.progress(current_step)
    for percent_complete in range(current_step, total_steps):
        time.sleep(0.01)  # 模拟处理时间
        progress_bar.progress(percent_complete + 1)
    st.success(text)

def process_text_message(user_msg, model_type,if_analysis_sentence,scenarios):
    overall_prediction = predict_model(user_msg,model_type,if_analysis_sentence,scenarios)
    return overall_prediction

def process_file(file, model_type,if_analysis_sentence,scenarios):
    if file.name.endswith('.txt'):
        text = read_texts_from_file(file)
        return process_texts(text, model_type,if_analysis_sentence,scenarios)
    if file.name.endswith('.wav'):
        return process_audio(file, model_type,if_analysis_sentence,scenarios)
    else:
        return process_image(file, model_type,if_analysis_sentence,scenarios)

def process_image(file, model_type,if_analysis_sentence,scenarios):
    text=ocr_test(0,file)
    return process_text_message(text, model_type,if_analysis_sentence,scenarios)

def read_texts_from_file(file):

    print("read_texts_from_file")
    text=file.read().decode('utf-8')
    return text

def process_texts(texts, model_type):
    model = PPL_LL_based_gpt2_t5_small()
    results = []
    total_texts = len(texts)
    for i, text in enumerate(texts):
        show_progress(text=f"正在处理第{i+1}篇文本...", current_step=(i * 100) // total_texts)
        result = model(text, model_type)
        results.append(result)
    return results


def ocr_test(pos,PIC_FILE):
    APP_ID = '3035921089'
    APP_KEY = 'XCjChmlRyvPeEmMz'
    DOMAIN = 'api-ai.vivo.com.cn'
    URI = '/ocr/general_recognition'
    METHOD = 'POST'
    URI_c = '/vivogpt/completions'
    URI_s = '/translation/query/self'
    b_image=PIC_FILE.read()
    image = base64.b64encode(b_image).decode("utf-8")
    post_data = {"image": image, "pos": int(pos), "businessid": "1990173156ceb8a09eee80c293135279"}
    """1990173156ceb8a09eee80c293135279，支持旋转图像、非正向文字识别

        8bf312e702043779ad0f2760b37a0806，只支持正向文字识别，耗时比1990小"""
    params = {}
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)

    url = 'http://{}{}'.format(DOMAIN, URI)
    response = requests.post(url, data=post_data, headers=headers)
    if response.status_code == 200:
        if pos == 0:
            print(response.json()['support'])
            text = ""
            for word in response.json()['result']['words']: #拼接得到的words，保持分段结构
                text += word['words'] + '\n'
                #delete "VIVO识图提供技术支持"
                text = text.replace("VIVO识图提供技术支持", "")
                print(text)
            return text
        elif pos == 2:
            return response.json()['result']
    else:
        print(response.status_code, response.text)

def process_audio(file, model_type):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            st.write("识别的文本是:", text)
            return process_text_message(text, model_type)
    except sr.UnknownValueError:
        st.error("Google Speech Recognition引擎无法理解音频")
    except sr.RequestError as e:
        st.error(f"Google Speech Recognition服务出现了错误; {e}")

def sentence_split(text):
    sentences = [s.strip() for s in re.split(r'[。.|]+', text) if s.strip()]
    return sentences

def highlight_sentences(sentences, model_type):
    results = []
    total_sentences = len(sentences)
    # 初始化进度条
    progress_bar = st.progress(0)
    for i, sentence in enumerate(sentences):
        prediction = process_text_message(sentence, model_type)
        ai_prob = prediction.get('ai_prob', 0.0)
        if ai_prob > 0.5:
            highlighted_sentence = f"<span class='highlighted-sentence'>{sentence}</span>"
            results.append((highlighted_sentence, ai_prob))
        else:
            results.append((sentence, ai_prob))
        # 更新进度条
        progress_bar.progress(((i + 1) * 100) // total_sentences)
    # 完成进度条
    progress_bar.progress(100)
    st.success('逐句检测完成')
    return results

def main():
    main_bg('./assets/background.png')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        st.image('assets/logo.png', width=220)
    set_page_style()
    st.title('NoGPT——AI生成文本检测平台')

    scenarios = st.multiselect('请选择检测场景', ['新闻场景', '论文场景','百科场景','文学场景','其他场景'])
    if scenarios==['其他场景']:
        scenarios="none"
    elif scenarios==['新闻场景']:
        scenarios="life&news"
    elif scenarios==['论文场景']:
        scenarios="paper"
    elif scenarios==['百科场景']:
        scenarios="science&tech"
    elif scenarios==['文学场景']:
        scenarios="literature"
    if not scenarios:
        st.warning("请选择至少一个检测场景")
        return

    detection_type = st.radio("选择检测类型", ["整体检测", "成分检测"])

    user_msg = st.text_area("👉输入您的消息：")

    uploaded_file = st.file_uploader("📂上传文件", type=["txt", "jpg", "png", "jpeg", "wav"])

    model_type = st.selectbox('🔑选择mask模型类型', ['t5-small', 't5-large','blueLM','deepseek', 'none'], index=0)

    start_detect = st.button('⏳开始检测')
    if start_detect:
        start_time = time.time()  # 记录开始时间
        if user_msg:
            print(detection_type)
            if detection_type == "整体检测":
                show_progress(text='正在进行整体检测...')
                #print(type(user_msg))
                overall_prediction = predict_model(user_msg,model_type,0,scenarios)
                showUp=""
                i=0
                for prediction in overall_prediction:
                    i+=1
                    print(prediction)
                    if prediction['Predicted']==1:
                        prediction['conclusion']=f"第{i}篇文本是由AI生成的"
                    else:
                        prediction['conclusion']=f"第{i}篇文本是由人类撰写的"
                    showUp+=f"{prediction['conclusion']}，人类概率：{prediction['Human Probability']}，AI概率：{prediction['AI Probability']} \n\n"
                st.markdown(f'<div class="pink-box"><b>整体文本检测结果：</b><br>'
                            f"{showUp}",
                            unsafe_allow_html=True)

            elif detection_type == "成分检测":
                
                overall_prediction = predict_model(user_msg,model_type,1,scenarios)

                st.markdown('<div class="pink-box"><b>成分检测结果如下:</b><br>', unsafe_allow_html=True)
                for i,prediction in enumerate(overall_prediction):

                    if prediction['Human Probability']>0.4 and prediction['Human Probability']<0.6:
                        prediction['Hman Probability']=prediction['Human Probability']+0.1
                        prediction['AI Probability']=prediction['AI Probability']-0.1
                    if prediction['AI Probability']>prediction['AI Probability']:
                        prediction['conclusion']=f"第{i}篇文本是由AI生成的"
                    else:
                        prediction['conclusion']=f"第{i}篇文本是由人类撰写的"
                    if prediction['AI Probability']>prediction['Human Probability']:
                       st.markdown(f'<div class="sentence-box">'
                        f"""<span class="highlighted-sentence">{prediction['sentence']}\n{prediction['conclusion']}(AI概率: <span class='bold-red'>{prediction['AI Probability']}</span> 人类概率： <span class='bold-red'>{prediction['Human Probability']} </span>)\n\n""", unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="sentence-box">'
                        f"""{prediction['sentence']}\n{prediction['conclusion']}(AI概率: <span class='bold-red'>{prediction['AI Probability']}</span> 人类概率： <span class='bold-red'>{prediction['Human Probability']}</span>)\n\n""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.success('成分检测完成')
        elif uploaded_file:
            if detection_type=="整体检测":
                show_progress(text='正在处理上传的文件...')
                results = process_file(uploaded_file, model_type,0,scenarios)
                showUp=""
                i=0
                for prediction in results:
                    i+=1
                    print(prediction)
                    if prediction['Predicted']==1:
                        prediction['conclusion']=f"第{i}篇文本是由AI生成的"
                    else:
                        prediction['conclusion']=f"第{i}篇文本是由人类撰写的"
                    showUp+=f"{prediction['conclusion']}，人类概率：{prediction['Human Probability']}，AI概率：{prediction['AI Probability']} \n"
                st.markdown(f'<div class="pink-box"><b>整体文本检测结果：</b><br>'
                            f"{showUp}",
                            unsafe_allow_html=True)
                st.markdown('<div class="pink-box"><b>成分检测结果如下:</b><br>', unsafe_allow_html=True)
            elif detection_type=="成分检测":
                show_progress(text='正在处理上传的文件...')
                results = process_file(uploaded_file, model_type,1,scenarios)
                st.markdown('<div class="pink-box"><b>成分检测结果如下:</b><br>', unsafe_allow_html=True)
                for i,prediction in enumerate(results):

                    if prediction['Human Probability']>0.4 and prediction['Human Probability']<0.6:
                        prediction['Hman Probability']=prediction['Human Probability']+0.1
                        prediction['AI Probability']=prediction['AI Probability']-0.1
                    if prediction['AI Probability']>prediction['AI Probability']:
                        prediction['conclusion']=f"第{i}篇文本是由AI生成的"
                    else:
                        prediction['conclusion']=f"第{i}篇文本是由人类撰写的"
                    if prediction['AI Probability']>prediction['Human Probability']:
                       st.markdown(f'<div class="sentence-box">'
                        f"""<span class="highlighted-sentence">{prediction['sentence']}\n{prediction['conclusion']}(AI概率: <span class='bold-red'>{prediction['AI Probability']}</span> 人类概率： <span class='bold-red'>{prediction['Human Probability']}</span>)""", unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="sentence-box">'
                        f"""{prediction['sentence']}\n{prediction['conclusion']}(AI概率: <span class='bold-red'>{prediction['AI Probability']}</span> 人类概率： <span class='bold-red'>{prediction['Human Probability']}</span>)""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.success('成分检测完成')
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        st.write(f"检测完成，耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
