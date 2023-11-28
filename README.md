# enhance-long 

## Description

This tool(enhance_long) aims to enhance the LLM's(LlaMa2) long context extrapolation capability in the lowest-cost approach, preferably without training, and can be used directly in the LLM inference phase. 

## How to Install 
```
pip install enhance-long
```


## How to Use
This is very simple to use, you just need to
```
import enhance_long
```

then package `enhance_long`  automatically replace llama2 network architeture, and recommended using [flashattention2](https://github.com/Dao-AILab/flash-attention) by setting the llama2 `config.json` file, which will significantly accelerate and save GPU memory in model predicting. 


``` json
# config.json
{
  "_flash_attn_2_enabled": True,
  ......
}
```
Of course if you want to use flashattention2 the premise is that your environment needs support.


## Added Tech

- [x] Dynamic NTK RoPE (alpha=2, inspired by [reddit talk](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases))
- [x] Consistent Dynamic NTK RoPE (created by [Consistent-DynamicNTKRoPE](https://github.com/NormXU/Consistent-DynamicNTKRoPE), implemetation fix by @MrYXJ)
- [x] LogN Attention (inspired by [熵不变性看Attention的Scale操作](https://kexue.fm/archives/8823),[Overcoming a Theoretical Limitation of Self-Attention ](https://openreview.net/forum?id=qc9O2EtrMI-))
- [ ] ......(More advanced technologies are being searched and developed)


## Result
We use [LongBench](https://github.com/THUDM/LongBench) to evaluate how enhace_long enhances LlaMa2 ablity of long context extrapolation.Due to the poor effect of the original Llama on Chinese, the comparative results are not very convincing. We only select dataset which language is English and average text length is longer than 4k in LongBench: 
- HotpotQA: The average text length is 9K
- MusiQue:  The average text length is 11k
- NarrativeQA: The average text length is 18K.
- GovReport: The average text length is 8k.
- QMSum: The average text length is 10k.
- TriviaQA: The average text length is 8K
- SAMsum: The average text length is 6.25K
- TREC: The average text length is 5K
- PassageRetrieval-en: The average text length is 9.2k.
- PassageCount: The average text length is is 11K
- RepoBench-P:  The average text length is is 4.2k.


The llama2's origin max context length is 4k(4096), so we expand max context length to 8k, 16k, and 32k and  compare the results of direct extrapolation (not using enhance-long) with using the enhance-long in the above dataset.


### Origin 4K 

| Model | HotpotQA | MusiQue |NarrativeQA |GovReport | QMSum | TriviaQA| SAMsum| TREC| PassageRetrieval-en|PassageCount|RepoBench-P |
|--|--|--|--|--|--|--| --|--|--|--|--|
| Llama2-7B-chat | 23.52 | 6.39| 15.86 | 19.6 | 19.38| 84.11| 41.44| 56| 8.17| 2.01| 51.71|


### Extrapolation 8K
| Model | HotpotQA | MusiQue |NarrativeQA |GovReport | QMSum | TriviaQA| SAMsum| TREC| PassageRetrieval-en|PassageCount|RepoBench-P |
|--|--|--|--|--|--|--| --|--|--|--|--|
| Llama2-7B-chat | 0.86 | 0| 0| 1.44 | 0.11| 9| 6.1| 11.5| 0| 0.07| 5.34|
| Llama2-7B-chat-enhance | **28.64** | 7.02 | **19.85** | **20.98** | **20.79** | **84.45**| **42.7**| **63**| 7.9 | 2.91| 48.85|


### Extrapolation 16K
| Model | HotpotQA | MusiQue |NarrativeQA |GovReport | QMSum | TriviaQA| SAMsum| TREC| PassageRetrieval-en|PassageCount|RepoBench-P |
|--|--|--|--|--|--|--| --|--|--|--|--|
| Llama2-7B-chat | 0.86 | 0| 0| 1.44 | 0.11| 9| 6.1| 11.5| 0| 0.07| 3.85|
| Llama2-7B-chat-enhance | 27.17 | 10.27 | 17.2 | 20.93 | 20| 80.41| 41.6| 56.5| **11.08**| **6.82**| 45.77|


### Extrapolation 32K
| Model | HotpotQA | MusiQue |NarrativeQA |GovReport | QMSum | TriviaQA| SAMsum| TREC| PassageRetrieval-en|PassageCount|RepoBench-P |
|--|--|--|--|--|--|--| --|--|--|--|--|
| Llama2-7B-chat | 0.86 | 0| 0| 1.44 | 0.11| 9| 6.1| 11.5| 0| 0.07| 3.71|
| Llama2-7B-chat-enhance | 24.52 | **10.39** | 6.87 |  | 20.13 | 69.32 | 38.15| 59 | 5.14| 4.86| |


### Conclusion
- Compare different input length  in Llama2-7B-chat, When original Llama2-7B-chat (without using enhance_long tech) maximum input length exceeds 4k, ts capability decreases significantl.
- Compare without using enhance_long tech with using, it is clear that this approach can be **significantly improved** without finetuning. Especially in PassageCount dataset result, it outperform other all models(including continue finetuing) in [LongBench](https://github.com/THUDM/LongBench)
- Compare with continue finetuneing in long context, now this（directly used in the inference）overall effect is **not so good**， it only allowing llama2 extrapolate to the 8k and 16k intervals works better.



