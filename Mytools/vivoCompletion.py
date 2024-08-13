# encoding: utf-8
import uuid
import time
import requests
from Mytools.auth_util import gen_sign_headers
import json
import re

# 请替换APP_ID、APP_KEY
APP_ID = '***'
APP_KEY = '******'
URI = '/vivogpt/completions'
DOMAIN = 'api-ai.vivo.com.cn'
METHOD = 'POST'


def sync_vivogpt():
    params = {
        'requestId': str(uuid.uuid4())
    }
    print('requestId:', params['requestId'])

    message = [{
        "role": "user",
        "content":"用写一篇400词左右的不一样的中文文章，主题和格式不限，每次回答随机换主题生成文章，只需要输出文章就行，不能和之前的主题一样"}
    ]
    data = {
        "messages": message,
        'model': 'vivo-BlueLM-TB',
        'sessionId': str(uuid.uuid4()),
        'extra': {
            'temperature': 0.96
        }
    }
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
    headers['Content-Type'] = 'application/json'

    start_time = time.time()
    url = 'https://{}{}'.format(DOMAIN, URI)
    response = requests.post(url, json=data, headers=headers, params=params)

    if response.status_code == 200:
        res_obj = response.json()
        #print(f'response:{res_obj}')
        if res_obj['code'] == 0 and res_obj.get('data'):
            content = res_obj['data']['content']
            #print(f'final content:\n{content}')
            return content
    else:
        print(response.status_code, response.text)
    end_time = time.time()
    timecost = end_time - start_time
    print('请求耗时: %.2f秒' % timecost)

def validate_json_format(json_string,n_expected):
    try:
        # 尝试加载 JSON 字符串
        if json_string==None:
            return False
        else:
            data = json.loads(json_string)

        # 检查 completion 是否为字典
        if not isinstance(data, dict):
            print(f'{data} 不是字典')
            return False
        
        # 检查 completion 的键值对是否符合 <extra_id_i> 格式
        for key in data:
            if not re.match(r'^<extra_id_\d+>$', key):
                print(f'{key} 不符合 <extra_id_i> 格式')
                return False
            
            # 检查每个键对应的值是否为字符串
            if not isinstance(data[key], str):
                print(f'{key} 的值不是字符串')
                return False
        
        # 检查 completion 中的键数量是否>= n_expected
        if len(data) < n_expected:
            print(f'{len(data)} <{n_expected} 个')
            return False

        return True
    
    except json.JSONDecodeError:
        # JSON 解析错误
        print('JSON 解析错误')
        return False

def write_file(content):
    with open('ai_en.txt', 'a', encoding='utf-8') as f:  # 改为追加模式 'a'
        f.write('<text>' + content + '</text>\n')  # 添加标签并换行


def useFunctionCall(word,num,recover=False):
    params = {
    'requestId': str(uuid.uuid4())
    }
    print('requestId:', params['requestId'])

    message_recover = [{
        "role": "system",
        "content": """Generate the masked word (2 to 3 single word) in the text by {%s} <extra_id_i> tokens. Text: <text></text>, Output all the generated masked content in restrict JSON format: 
        {"<extra_id_0>": "Completion content 1", "<extra_id_1>": "Completion content 2", "<extra_id_2>": "Completion content 3", ...,"<extra_id_%s>":"Completion %s"} it must be %s elements in this format.
        EXAMPLE INPUT: 
        <text>Which is the <extra_id_0> mountain in the <extra_id_1>? Mount Everest.</text>

        EXAMPLE JSON OUTPUT:
        {
                "<extra_id_0>": "highest",
                "<extra_id_1>": "world"}
            }""" % (num,num-1,num,num)},           
        {
        "role": "user",
        "content": f"remember the correct JSON format,there are {num} <extra_id_i> in the text, you output wrong last time!<text>{word}</text>"
        }
    ]

    message = [{
        "role": "system",
        "content": """Generate the masked word (2 to 3 single word) in the text by {%s} <extra_id_i> tokens. Text: <text></text>, Output all the generated masked content in restrict JSON format: 
        {"<extra_id_0>": "Completion content 1", "<extra_id_1>": "Completion content 2", "<extra_id_2>": "Completion content 3", ...,"<extra_id_%s>":"Completion %s"} it must be %s elements in this format.
        EXAMPLE INPUT: 
        <text> 世界上 <extra_id_0> 山是哪一座? <extra_id_1>.</text>

        EXAMPLE JSON OUTPUT:
        {
                "<extra_id_0>": "最高的",
                "<extra_id_1>": "喜马拉雅山"}
            }
        your reply can not be none""" % (num,num-1,num,num)},           
        {
        "role": "user",
        "content": f"<text>{word}</text>"
        }
    ]
    data = {
        "messages": message_recover if recover else message,
        'model': 'vivo-BlueLM-TB',
        'sessionId': str(uuid.uuid4()),
        'extra': {
            'temperature': 0.98,
            'top_k': 50,
            'max_new_tokens': 512,
        }
    }
    headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
    headers['Content-Type'] = 'application/json'


    url = 'https://{}{}'.format(DOMAIN, URI)
    response = requests.post(url, json=data, headers=headers, params=params)

    if response.status_code == 200:
        res_obj = response.json()
        #print(f'response:{res_obj}')
        if res_obj['code'] == 0 and res_obj.get('data'):
            content = res_obj['data']['content']
            print(f'final content:\n{content}')
            return content
    else:
        print(response.status_code, response.text)


if __name__ == '__main__':

    text="""朝夕更替，日月流转。<extra_id_0>滚滚向前，时代的画卷不停变换无数伟人为我们书写下前进的注脚。前道不止，天地乃广。无论是过去还是现在，都是大有可为的时代，每个时代的画卷，都需要有为之士在其上书写下奋斗的篇章。已识乾坤大，长明火烛光。大有可为的时代为人提供了广阔的舞台，立志有为的人将在舞台上演绎精彩。

　　萤火微凉，光芒不减。我们什么时候拥有了喊出“这是一个大有可为的时代”的勇气？是100年前。在<extra_id_1>那段近乎黑暗的岁月里，压迫和屈辱笼罩着神州大地，中国差一点就失去了觉醒的希望。是那批最先觉醒的仁人志士的嘶吼创造出历史的新篇章，为中国带来了永恒光明的未来：李大钊、夏明翰、方志敏……他们前仆后继，以身许国，用他们的微光照亮了中华民族昏暗的前路，用他们的演讲回应时代的召唤，用他们的壮烈牺牲谱写“有为”的华章。“虽千万人吾往矣”，他们以肉身的陨灭、精神的长存告诉后人，“可为”的时代已然来临。

　　炬火炎炎，历久不灭。当“可为“成为时代的底气，“有为”的先驱便如雨后春笋般涌现。冲破重重阻拦回国的钱学森，在<extra_id_2>极其匮乏的情况下，排除万难，造出了中国自己的导弹和火箭，为祖国的国防事业做出了巨大贡献:放弃英国优越工作和生活条件的黄大年，为民族振兴不惜以身许国,为我国教育科研事业做出了突出贡献。他们将“有为”作为个人奋斗的原动力，在可为的时代做有为的人，助力国家发展，促进国家科技进步。中华民族复兴的伟大征程，被他们以科技报国留下浓墨重彩的一笔。

　　灼灼之光，传承不息。作为新时代青年，先辈们留下的<extra_id_3>已经深深融入我们的血液和灵魂之中。“可为”是让我们以坚定的信念和意志，拥抱时代提供的机遇；“有为"是让我们以积极奋斗的精神，勇担时代赋子我们的使命，共同建设伟大的祖国。我们何其有幸，生在无数先辈抛头颅洒热血为我们创造的美好时代中!现在正是我们回报先辈,接过历史接力棒去付出、去奉献的大好时候。新时代需要“有为”青年，圆梦民族复兴需要我们每个人的参与。

　　当历史的滚滚浪涛冲刷至我辈青年之时，我们不惧蜚语，心怀梦想，<extra_id_4>，剑指远方，发出青春最强音。我们心里有阳光，脚下有力量，大有可为的时代给了我们充分的支持，让我们去有所作为。
    """
    content = useFunctionCall(text)