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

    def test_llwithLength(self,texts):
        lls=[]
        lengths=[]
        print("perturbed ll ")
        plls=[self.get_ll(x) for x in self.perturb_texts_(texts, self.span_length, self.pct)]
        for text in texts:
            lls.append(self.get_ll(text))
            lengths.append(len(text))
        for ll,pll,length in zip(lls,plls,lengths):
            print(f"ll: {ll} pll: {pll} length: {length} dll: {ll-pll}")
        
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
            LL_Per_line = []
            PLL_Per_line = []
            DLL_Per_line = []
            Score_Per_line = []
            length_Per_line = []
            PPL_Per_line = []
            for sentence in sentences:
                length_Per_line.append(len(sentence))
                ll,pll=self.calc_ll_result(sentence)
                if pll=="error":
                    print("error")
                    continue
                pll=np.array(pll)
                mean_perturbed_ll=np.mean(pll)
                ppl=self.getPPL(sentence)
                
                PPL_Per_line.append(ppl)
                LL_Per_line.append(ll)
                PLL_Per_line.append(mean_perturbed_ll)
                DLL_Per_line.append(ll-mean_perturbed_ll)
                if mean_perturbed_ll-ll!=0:
                    std_perturbed_ll=np.std(pll)
                    if std_perturbed_ll!=0:
                        Score_Per_line.append((ll-mean_perturbed_ll)/std_perturbed_ll)
                    else:
                        Score_Per_line.append(999)
                else:
                    Score_Per_line.append(0)
            end_time=time.time()
            for ll,dll,score,ppl,length,ppl_per_line,sentence in zip(LL_Per_line,DLL_Per_line,Score_Per_line,PPL_Per_line,length_Per_line,PPL_Per_line,sentences):
                print(f"LL: {ll} D_LL: {dll} Score: {score} PPL per line: {ppl_per_line} Length:{length}")
                result=OrderedDict()
                result["LL"]=ll
                if dll==float('nan'):
                    result["D_LL"]=float('inf')
                    score=999
                else:
                    result["D_LL"]=dll
                    result["Score"]=score
                result["Perplexity"]=ppl_per_line
                result["length"]=length
                result['timecost']=end_time-start_time
                result['perturb_timecost']=sum(self.perturb_timecost)/len(self.perturb_timecost)
                result["perturbations"]=self.num_perturb
                result['base_model']=self.model_id
                result['mask_model']=self.mask_model_id
                result['sentence']=sentence
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
