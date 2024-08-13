def read_and_merge_texts_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    texts = content.split('</text>')
    texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
    print(len(texts))
    # 过滤掉少于100个词的文本
    filtered_texts = [text for text in texts if len(text.split()) >= 100]
    
    # 将少于100个词的文本每5篇合并，并用<text>和</text>包装
    short_texts = [text for text in texts if len(text.split()) < 100]
    if short_texts:
        merged_short_texts = []
        for i in range(0, len(short_texts), 5):
            merged_text = ' '.join(short_texts[i:i+5])
            merged_text = f'<text>{merged_text}</text>'  # 包装合并后的文本
            merged_short_texts.append(merged_text)
        
        # 将合并后的短文本添加到过滤后的长文本列表中
        filtered_texts.extend(merged_short_texts)
    
    # 写入新文件
    with open('new_ai.txt', 'w', encoding='utf-8') as file:
        for text in filtered_texts:
            file.write(text + '</text>\n')  # 每个文本后添加结束标签和换行符

    return filtered_texts

texts = read_and_merge_texts_from_file('dataset/new_ai.txt')
print(len(texts))  # 输出合并后的文本数量