#### GPTMINUS 使用文档

base PATH:`GPTMINUS/`

数据集路径：`dataset/`

预测matirx路径：`matirx/`

​	|其中，cache_matirx存放每次运行结果，与实际投入训练的matrix隔离，方便控制质量

文本处理，遮蔽模型API调用相关工具：`Mytools`

前端人员信息补充页：`pages`

svm训练，使用，模型存放：`svm_model`

所有训练，预测信息保存留档：`result.txt`



大模型下载路径：GPTMINUS同级目录的`models/`

```
GPTMINUS/ #project
models/ #model path
```

我们的数据集：

联系我们获取 Baoquan_Cao.outlook.com

##### 本地部署流程

1 下载GPTMINUS项目

2 使用数据集试验出一批矩阵,区分中英文,如有特殊需求可以按情景分类

3 使用实验得到的分布数据训练出一个对应svm模型存入svm_model下,修改run_new中的读取模型的路径

4 启动web使用

##### matrix实验数据

使用train4data.py 文本路径 base_model_id  mask_model_id if_analysis_sentence perturbations label env recover 运行

模型id及env映射表:

```
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
```



##### web启动运行平台

streamlit run webUI.py



##### 注意事项

英文对应遮蔽文本: t5-large / t5-small

 中文对应遮蔽文本: blueLM / deepseek (需要API key 填入Mytools/vivoCompletion.py & deepseekCompletion.py 下)

目前成分分析仅支持中文,deepseek

一次输入多篇文本需要使用`<text> </text>`做分隔符,暂不支持中英文混篇输入