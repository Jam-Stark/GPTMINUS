import re

def clean_text(text):
    # 删除所有换行符、英文字符和空格
    text = re.sub(r'[\r\n]', '', text)  # 删除换行符
    text = re.sub(r'[a-zA-Z]', '', text)  # 删除英文字符
    text = re.sub(r'\s', '', text)  # 删除空格
    return text

def insert_tags(text):
    words = list(text)
    chunks = [''.join(words[i:i+400]) for i in range(0, len(words), 400)]
    return ''.join(f'<text>{chunk}</text>' for chunk in chunks)

def split_texts(content):
    # 找到所有的<text></text>标签内容
    pattern = re.compile(r'<text>(.*?)</text>', re.DOTALL)
    texts = pattern.findall(content)

    # 处理每个文本块
    processed_texts = []
    for text in texts:
        cleaned_text = clean_text(text)
        if len(cleaned_text) >= 100:  # 丢弃不足100词的文本
            processed_texts.append(insert_tags(cleaned_text))

    # 重新拼接处理过的文本块
    new_content = ''.join(processed_texts)

    return new_content