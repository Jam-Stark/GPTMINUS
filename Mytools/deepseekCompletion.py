import json
from openai import OpenAI


def deepseekCompletion(text,n_expected,_recover=False):
    client = OpenAI(
        api_key="********",
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
    The user will provide a masked text with <extra_id_\d+>. Please simply parse the "<extra_id_\d+>"  and output them in JSON format. 

    EXAMPLE INPUT: 
    Which is the <extra_id_0> mountain in the <extra_id_1>? Mount Everest.

    EXAMPLE JSON OUTPUT:
    {
        "<extra_id_0>": "highest",
        "<extra_id_1>": "world"
    }
    the completion each should not more than 3 word!
    """

    recovery_prompt = f"""remember the correct JSON format,there are {n_expected} <extra_id_i> in the text, you output wrong last time!<text>{text}</text> """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": text if not _recover else recovery_prompt}]

    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    #print(json.loads(response.choices[0].message.content))

    return response.choices[0].message.content