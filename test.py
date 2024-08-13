#对中文文本的检测方法Implement
import torch
import numpy as np
import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import GPT2LMHeadModel, GPT2Tokenizer,T5ForConditionalGeneration, AutoTokenizer,AutoModelForMaskedLM,AutoModelForCausalLM,AutoModelForSeq2SeqLM
from collections import OrderedDict
import jieba
import concurrent.futures

from Mytools import vivoCompletion
from Mytools import deepseekCompletion
import json
import time

n_positions = 1024  # 假设的模型最大长度，根据实际情况调整
"""
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-72B-Chat",
    torch_dtype="auto",device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat")

IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese

vivo-ai/BlueLM-7B-Base
"""
class PPL_LL_based_cn:
    def __init__(self, device="cuda", model_id="qwen"):
        self.device = device
        self.model_id = model_id
        self.perturb_timecost = []

        #self.mask_tokenizer = AutoTokenizer.from_pretrained("mxmax/Chinese_Chat_T5_Base",cache_dir='../models',legacy=False)
        #self.mask_model = AutoModelForSeq2SeqLM.from_pretrained("mxmax/Chinese_Chat_T5_Base",cache_dir='../models').to(device)

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            torch_dtype="auto",
            device_map="auto",cache_dir='../models'
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct",cache_dir='../models')
        #self.tokenizer = AutoTokenizer.from_pretrained("vivo-ai/BlueLM-7B-Base",cache_dir='../models', trust_remote_code=True, use_fast=False)
        #self.model = AutoModelForCausalLM.from_pretrained("vivo-ai/BlueLM-7B-Base",cache_dir='../models', device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
        
        #self.model = GPT2LMHeadModel.from_pretrained(model_id,cache_dir='../models').to(device)
        #self.tokenizer = GPT2Tokenizer.from_pretrained(model_id,cache_dir='../models')

        #self.mask_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese',cache_dir='../models',padding_side='left')
        #self.mask_model = AutoModelForMaskedLM.from_pretrained('google-bert/bert-base-chinese',cache_dir='../models').to(device)

        self.buffer_size = 5
        self.mask_top_p = 0.96
        self.span_length = 2 # 每个遮蔽词汇的长
        self.pct = 0.2  # 需要被遮蔽的词汇所占的比例
        self.top_k=40

        self.num_perturb=15 #单个文本获得的扰动文本数量，数量越大，计算时间越长，但平均值越准确

        #self.max_length = self.model.config.n_positions
        self.max_length = 1024
        self.stride = 51

    def tokenize_and_mask(self,text, span_length, pct, ceil_pct=False):
        tokens = list(jieba.cut(text, cut_all=False))
        mask_string = '<<<mask>>>'

        n_spans = pct * len(tokens) / (span_length + self.buffer_size * 2)
        #print("n_spans: ",n_spans)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)
        n_spans = min(n_spans, len(tokens) - span_length)
        #print("n_spans: ",n_spans)
        n_masks = 0
        if n_spans == 0:
            n_spans = 1
        while n_masks < n_spans:
            #print("进入循环")
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)

            """chinese_punctuation = "，。！？；：、（）《》“”‘’"
            #tokens[search_start:search_end]中不包含中文标点符号
            span_contains_punctuation = any(char in chinese_punctuation for char in ''.join(tokens[search_start:search_end]))"""

            if mask_string not in tokens[search_start:search_end]:
                #print("应该添加mask")
                tokens[start:end] = [mask_string]
                #print(tokens[start:end])
                n_masks += 1
        #print("tokens: ",tokens)
        #print("n_masks: ",n_masks)
        #暂停等待
        #input("Press Enter to continue...")
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        # 对分词后的tokens重新拼接成句子
        text = ''.join(tokens)
        ##print("after apply id in text: ",text)
        return text, n_masks,tokens

    def count_masks(self,texts):
        return [len([x for x in text if x.startswith("<extra_id_")]) for text in texts]
    
    def vivoGenMasks(self, texts,n_expected):
        masks = []
        #n_expected = self.new_count_masks(texts)
        #print("n_expected: ",n_expected)
        # 使用 zip 将 texts 和 n_expected 打包在一起
        for text, num in zip(texts, n_expected):
            #print("num:", num)
            mask = vivoCompletion.useFunctionCall(text, num)
            #check if mask is a json string
            retry_count=0
            while not vivoCompletion.validate_json_format(mask,num):
                print("JSON format error, retrying...")
                mask = vivoCompletion.useFunctionCall(text, num,recover=True)
                retry_count+=1
                if retry_count>5:
                    print("retry too many times, exit")
                    mask="error"
                    break

            if mask!="error":
                mask = json.loads(mask)
                print("fiils generated")
            #print(mask["completion"])
            masks.append(mask)
        
        return masks

    def vivoGenMasks_single(self, text,n_expected):
        #print("num:", num)
        mask = vivoCompletion.useFunctionCall(text, n_expected)
        #check if mask is a json string
        retry_count=0
        while not vivoCompletion.validate_json_format(mask,n_expected):
            print("JSON format error, retrying...")
            mask = vivoCompletion.useFunctionCall(text, n_expected,recover=True)
            retry_count+=1
            if retry_count>5:
                print("retry too many times, exit")
                mask="error"
                break

        if mask!="error":
            mask = json.loads(mask)
            print("fiils generated")
        
        return mask

    def vivoGenMasks_parallel(self, texts,n_expected):
        masks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            masks = executor.map(self.vivoGenMasks_single,texts,n_expected)
        return masks

    def deepseekGenMasks(self, texts,n_expected):
        masks = []
        #n_expected = self.new_count_masks(texts)
        #print("n_expected: ",n_expected)
        # 使用 zip 将 texts 和 n_expected 打包在一起
        for text, num in zip(texts, n_expected):
            #print("num:", num)
            mask = deepseekCompletion.deepseekCompletion(text)
            #check if mask is a json string
            retry_count=0
            while not vivoCompletion.validate_json_format(mask,num):
                print("JSON format error, retrying...")
                mask = deepseekCompletion.deepseekCompletion(text, num,recover=True)
                retry_count+=1
                if retry_count>5:
                    print("retry too many times, exit")
                    mask="error"
                    break

            if mask!="error":
                mask = json.loads(mask)
                print("fiils generated")
            #print(mask["completion"])
            masks.append(mask)
        
        return masks

    def deepseekGenMasks_single(self, text,n_expected):
        #print("num:", num)
        mask = deepseekCompletion.deepseekCompletion(text,n_expected,_recover=False)
        #check if mask is a json string
        retry_count=0
        while not vivoCompletion.validate_json_format(mask,n_expected):
            print("JSON format error, retrying...")
            mask = deepseekCompletion.deepseekCompletion(text,n_expected,_recover=True)
            retry_count+=1
            if retry_count>5:
                print("retry too many times, exit")
                mask="error"
                break

        if mask!="error":
            mask = json.loads(mask)
            print("fiils generated")
        
        return mask
    def deepseekGenMasks_parallel(self, texts,n_expected):
        masks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            masks = executor.map(self.deepseekGenMasks_single,texts,n_expected)
        #关闭线程池
        
        return masks


    def apply_vivo_fills(self,masked_tokens, vivo_fills,n_expecteds):
        #print(len(masked_tokens))
        for idx, (token, fills, n) in enumerate(zip(masked_tokens, vivo_fills, n_expecteds)):
            #print("fills: ",fills)
            #print(type(fills))
            #print(fills["<extra_id_1>"])
            if len(fills) < n:
                token = []
            else:
                for fill_idx in range(n):
                    #print(f"<extra_id_{fill_idx}>")
                    #print(token)
                    token[token.index(f"<extra_id_{fill_idx}>")] = fills[f"<extra_id_{fill_idx}>"]
        # join tokens back into text
        texts = ["".join(x) for x in masked_tokens]
        #sprint(texts)
        return texts

    def perturb_texts_(self,texts, span_length, pct, ceil_pct=False):
            masked_texts=[] 
            num_masks = []
            masked_tokenss=[]
            for x in texts:
                masked_text,num_mask,masked_tokens=self.tokenize_and_mask(x, span_length, pct, ceil_pct)
                masked_texts.append(masked_text)
                num_masks.append(num_mask)
                masked_tokenss.append(masked_tokens)
            #print(len(masked_tokenss))
            print("mask texts done")
            start_time = time.time()
            if self.mask_model_id=="blueLM":
                fills = self.vivoGenMasks(masked_texts,num_masks)

            elif self.mask_model_id=="deepseek":
                fills = self.deepseekGenMasks_parallel(masked_texts,num_masks)

            else:
                print("mask_model_id error")
                exit(0)
            end_time = time.time()
            perturb_timecost = end_time - start_time
            print('fill mask %s 耗时: %.2f秒' % (self.mask_model_id,perturb_timecost))
            self.perturb_timecost.append(perturb_timecost)
            perturbed_texts = self.apply_vivo_fills(masked_tokenss, fills, num_masks)

            return perturbed_texts

    def get_ll(self,text):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
            labels = tokenized.input_ids
            return -self.model(**tokenized, labels=labels).loss.item()
    
    def calc_ll_result(self,text):
        #return self.get_ll(text)-self.get_perturbed_ll(text)
        text_for_perturbed_ll=[]
        i=0
        for(i) in range(self.num_perturb):
            text_for_perturbed_ll.append(text)
        #return self.get_ll(text),self.get_multi_perturbed_ll(text_for_perturbed_ll)
        #print(self.perturb_texts_(text_for_perturbed_ll, self.span_length, 0.15))
        plls=[]
        for x in self.perturb_texts_(text_for_perturbed_ll, self.span_length, self.pct):
            if x=="error":
                print("skip")
                continue
            plls.append(self.get_ll(x))
        return self.get_ll(text),plls
        
    def __call__(self, text,if_analysis_sentences,mask_model_id="blueLM",n_perturbations=15):
        start_time=time.time()
        results = OrderedDict()
        self.mask_model_id=mask_model_id
        self.num_perturb=n_perturbations
        results["length"]=len(text)
        results["mask_model"]=self.mask_model_id
        results["base_model"]=self.model_id

        #将text根据中文句号拆开
        sentences = re.split(r'(。|？|！|……)', text)
        # 将标点符号重新附加到各个句子末尾
        sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]
        print("if_analysis_sentences: ",if_analysis_sentences)
        if if_analysis_sentences==1:
            results=[]
            self.num_perturb=10
            self.span_length=3
            self.pct=0.15
            print("Analysis sentences")

            len_num=len(sentences)
            
            window_size=4
            print("window_size: ",window_size)
            slide_size=2
            print("slide_size: ",slide_size)

            i=0
            lls=[list() for _ in range(len_num)]
            pll=[list() for _ in range(len_num)]
            dlls=[list() for _ in range(len_num)]
            scores=[list() for _ in range(len_num)]
            ppls=[list() for _ in range(len_num)]
            while i+window_size<=len_num:
                print("i: ",i)
                #拼接window_size个句子
                perturbed_sentences="".join(sentences[i:i+window_size])
                ll,plls=self.calc_ll_result(perturbed_sentences)
                #print(f"perturbed LL: {plls} ll: {ll}")

                if plls=="error":
                    print("error")
                    continue
                plls=np.array(plls,dtype=np.float64)
                #print(f"plls: {plls}")
                mean_perturbed_ll=np.mean(plls)
                dll=ll-mean_perturbed_ll
                if dll!=0:
                    std_perturbed_ll=np.std(plls)
                    #print(f"std_perturbed_ll: {std_perturbed_ll}")
                    if std_perturbed_ll!=0:
                        score=(ll-mean_perturbed_ll)/std_perturbed_ll
                    else:
                        score=999
                else:
                    score=0
                
                ppl=self.getPPL(perturbed_sentences)
                #input("Press Enter to continue...")
                for j in range(i,i+window_size):
                    lls[j].append(ll)
                    dlls[j].append(dll)
                    scores[j].append(score)
                    ppls[j].append(ppl)
                    pll[j].append(mean_perturbed_ll)
            
                i+=slide_size
            
            #处理剩余的句子
            if i<len_num:
                perturbed_sentences="".join(sentences[i:])
                ll,plls=self.calc_ll_result(perturbed_sentences)
                if plls=="error":
                    print("error")
                    return
                plls=np.array(plls)
                mean_perturbed_ll=np.mean(plls)
                dll=ll-mean_perturbed_ll
                if dll!=0:
                    std_perturbed_ll=np.std(plls)
                    if std_perturbed_ll!=0:
                        score=(ll-mean_perturbed_ll)/std_perturbed_ll
                    else:
                        score=999
                else:
                    score=0
                ppl=self.getPPL(perturbed_sentences)
                for j in range(i,len_num):
                    lls[j].append(ll)
                    dlls[j].append(dll)
                    scores[j].append(score)
                    ppls[j].append(ppl)
                    pll[j].append(mean_perturbed_ll)
            
            #追加开头2句
            for j in range(0,slide_size):
                ll,plls=self.calc_ll_result(sentences[j])
                if plls=="error":
                    print("error")
                    return
                plls=np.array(plls)
                mean_perturbed_ll=np.mean(plls)
                dll=ll-mean_perturbed_ll
                if dll!=0:
                    std_perturbed_ll=np.std(plls)
                    if std_perturbed_ll!=0:
                        score=(ll-mean_perturbed_ll)/std_perturbed_ll
                    else:
                        score=999
                else:
                    score=0
                ppl=self.getPPL(perturbed_sentences)
                for j in range(0,slide_size):
                    lls[j].append(ll)
                    dlls[j].append(dll)
                    scores[j].append(score)
                    ppls[j].append(ppl)
                    pll[j].append(mean_perturbed_ll)

            #追加窗口滑动末尾2句
            for j in range(i-2,i):
                ll,plls=self.calc_ll_result(sentences[j])
                if plls=="error":
                    print("error")
                    return
                plls=np.array(plls)
                mean_perturbed_ll=np.mean(plls)
                dll=ll-mean_perturbed_ll
                if dll!=0:
                    std_perturbed_ll=np.std(plls)
                    if std_perturbed_ll!=0:
                        score=(ll-mean_perturbed_ll)/std_perturbed_ll
                    else:
                        score=999
                else:
                    score=0
                ppl=self.getPPL(perturbed_sentences)
                for j in range(len_num-slide_size,len_num):
                    lls[j].append(ll)
                    dlls[j].append(dll)
                    scores[j].append(score)
                    ppls[j].append(ppl)
                    pll[j].append(mean_perturbed_ll)
            end_time=time.time()

            for sentence,ll,dll,score,ppl,pl in zip(sentences,lls,dlls,scores,ppls,pll):
                result=OrderedDict()
                print(f"LL: {ll}")
                if dll==float('nan'):
                    result["D_LL"]=float('inf')
                    score=999
                else:
                    result["D_LL"]=dll
                    result["Score"]=score
                print(f"Score: {score}")
                print(f"Perplexity: {ppl}")
                print(f"Perturbed LL: {pl}")
                result["LL"]=np.mean(np.array(ll))
                result["D_LL"]=np.mean(np.array(dll))
                result["Score"]=np.mean(np.array(score))
                result["Perplexity"]=np.mean(np.array(ppl))
                result["Perturbed LL"]=np.mean(np.array(pl))
                result["timecost"]=end_time-start_time
                result["sentence"]=sentence
                results.append(result)
            return results
        else:
            print("Analysis text")
            results["perturbations"]=self.num_perturb
            text_for_ll=text
            ll,perturbed_ll=self.calc_ll_result(text_for_ll)

            #print(f"perturbed LL: {perturbed_ll} ll: {ll}")
            perturbed_ll=np.array(perturbed_ll)
            mean_perturbed_ll=np.mean(perturbed_ll)
            if mean_perturbed_ll-ll!=0:
                std_perturbed_ll=np.std(perturbed_ll)
                if std_perturbed_ll!=0:
                    results["D_LL"] = ll-mean_perturbed_ll
                    results["Score"]=results["D_LL"]/std_perturbed_ll
                else:
                    results["D_LL"] = ll-mean_perturbed_ll
                    results["Score"]=999
            else:
                results["D_LL"] = 0
                results["Score"]=0

            results["LL"] = ll
            results["Perturbed LL"] = mean_perturbed_ll

            print(f" base_model: {results['base_model']} mask_model: {results['mask_model']} perturbations: {results['perturbations']}")


            ppl = self.getPPL(text)
            #print(f"Perplexity {ppl}")
            results["Perplexity"] = ppl



            PPL_Per_line = [self.getPPL(x) for x in sentences]

            results["Perplexity per line"] = sum(PPL_Per_line) / len(PPL_Per_line)

            print(f"LL: {results['LL']} D_LL: {results['D_LL']} Score: {results['Score']} PPL: {results['Perplexity']}, PPL_per_line: {results['Perplexity per line']} length: {results['length']} ")

            end_time=time.time()
            results['timecost']=end_time-start_time
            results['perturb_timecost']=sum(self.perturb_timecost)/len(self.perturb_timecost)
            return results

    def getPPL(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        likelihoods = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                if torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
                    print(f"Encountered nan or inf in neg_log_likelihood at segment {begin_loc} to {end_loc}")
                    continue
                likelihoods.append(neg_log_likelihood)
                #print("neg_log_likelihood: ", neg_log_likelihood)
                nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if len(nlls) == 0:
            print("All segments resulted in nan or inf. Returning inf as PPL.")
            return float('inf')

        #print("nlls: ", nlls)
        #print("end_loc: ", end_loc)
        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl


#test
model_cn=PPL_LL_based_cn(model_id="qwen")
text=""" <text>随着科技的飞速发展，人工智能（AI）已经成为推动社会进步的重要力量。AI技术的应用范围日益扩大，从智能语音助手到自动驾驶汽车，从医疗诊断到金融分析，AI正在改变我们的生活方式和工作方式。本文将探讨AI技术的最新进展及其对未来社会的影响。

首先，AI技术的核心在于机器学习，特别是深度学习。深度学习通过模拟人脑神经网络的工作原理，使计算机能够从大量数据中学习和提取特征，从而实现复杂的任务。近年来，随着计算能力的提升和大数据的普及，深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。

在医疗领域，AI技术的应用正在改变疾病的诊断和治疗方式。例如，通过分析患者的医学影像数据，AI系统可以辅助医生更准确地诊断癌症等疾病。此外，AI还可以帮助医生制定个性化的治疗方案，提高治疗效果。在金融领域，AI技术被用于风险评估、投资决策和欺诈检测，提高了金融服务的效率和安全性。

然而，AI技术的发展也带来了一系列挑战和问题。首先，随着AI系统的普及，数据隐私和安全问题日益突出。如何保护个人数据不被滥用，是当前亟待解决的问题。其次，AI技术的应用可能导致部分工作岗位的消失，对就业市场产生影响。因此，政府和企业需要制定相应的政策和措施，帮助受影响的员工转型和再就业。

此外，AI技术的伦理问题也不容忽视。例如，自动驾驶汽车在面临事故时如何做出决策，是一个复杂的伦理问题。如何在保证安全的同时，确保AI系统的决策符合人类的价值观和道德标准，是未来AI技术发展中需要重点考虑的问题。

总之，AI技术的发展为社会带来了巨大的机遇和挑战。我们需要在推动技术进步的同时，关注其对社会、经济和伦理的影响，确保AI技术的发展能够造福全人类。未来，随着技术的不断进步和应用场景的拓展，AI将继续在各个领域发挥重要作用，推动社会向更加智能化、高效化的方向发展。</text>
"""
results=model_cn(text,mask_model_id="deepseek",if_analysis_sentences= 1,n_perturbations=15)