Dưới đây là toàn bộ nội dung của bài báo được tổng hợp và trình bày lại dưới định dạng Markdown, bao gồm đầy đủ các phần từ Tiêu đề, Tóm tắt (Abstract) cho đến Phụ lục (Appendix) dựa trên dữ liệu bạn đã cung cấp:

```markdown
# Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs

**Authors:** Yuchen Fu, Zifeng Cheng, Zhiwei Jiang, Zhonghui Wang, Yafeng Yin, Zhengliang Li, Qing Gu.

## Abstract
Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token. However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token. To this end, we propose a novel Token Prepending (TP) technique that prepends each layer’s decoded sentence embedding to the beginning of the sentence in the next layer’s input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism. The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs. Extensive experiments demonstrate that our proposed TP technique can significantly improve the performance of existing prompt-based sentence embedding methods across different LLMs, while incurring negligible additional inference cost.

---

## 1. Introduction
Sentence embeddings have a wide range of applications in real-world scenarios, such as information retrieval, recommender systems, sentiment analysis, document clustering, and so on. With the success of LLMs in zero-shot settings for various NLP tasks, researchers focus on directly extracting sentence embeddings from LLMs without additional fine-tuning. This training-free setup is practical as it does not require training data, avoids fine-tuning costs, and prevents the potential loss of general semantic understanding.

Unlike bidirectional models like BERT, current LLMs are mostly decoder-only models with causal attention, which make the earlier tokens in the sentence cannot attend to the latter tokens. Recent studies attempt to prompt the model to encode sentence information into the embedding of the last token (i.e., the `<SET>`), which can attend to all preceding tokens. However, even if the last token is able to attend to all tokens, the earlier tokens still cannot attend to the later tokens (backward dependency), resulting in biased encoding and cascading effects on the last token. 

In this paper, we propose a simple yet effective technique called Token Prepending (TP). Our core idea is to prepend each layer’s decoded sentence embedding to the beginning of the sentence in the next layer’s input, allowing earlier tokens to attend to the complete sentence information. TP is entirely training-free and introduces no additional learnable parameters. We also propose an early-exit strategy that outputs embeddings from intermediate layers, rather than the final layer, to serve as sentence embeddings.

**Main contributions:**
* We propose a novel TP technique for eliciting sentence embeddings from LLMs that can be seamlessly integrated with various prompt-based methods with minimal additional inference overhead.
* We perform an in-depth exploration of the TP technique, including the optimal layer scope and the early exit strategy.
* Extensive experiments on STS benchmarks and downstream classification tasks demonstrate that TP significantly improves existing prompt-based methods.

---

## 2. Related Work
**Sentence Embeddings:** Previous research often employs unsupervised or supervised contrastive learning to fine-tune smaller pre-trained models (e.g., Sentence-T5) to enhance sentence embeddings. We focus on embeddings extracted by LLMs without fine-tuning.

**LLMs for Sentence Embeddings:** Some studies focus on enhancing sentence embeddings of LLMs through fine-tuning, mostly replacing unidirectional attention with bidirectional attention (e.g., BeLLM). However, fine-tuning is expensive and results in the loss of general capabilities.

**Extracting Sentence Embeddings from LLMs:** Existing methods focus on designing prompts, such as PromptEOL, Echo embeddings (repeating the input twice), MetaEOL, Pretended CoT, and Knowledge Enhancement. We propose the plug-and-play TP technique to improve these various prompt-based methods.

---

## 3. Preliminary
Previous work mainly focused on eliciting sentence embeddings through prompt engineering without interfering with internal operations. For example, PromptEOL introduces the template: `This sentence: “[Text]” means in one word: “`. The phrase “in one word” limits a sentence to being represented by the embedding of a single word, decoding the Sentence Embedding Token (SET). The previous work uses the last layer’s hidden state for the sentence embedding token as the output sentence embedding.

---

## 4. Proposed Method

### 4.1 Overview
Our core idea is to prepend the decoded sentence embedding token from the previous layer to the sentence in the next layer’s input. We perform the TP operation within the layer scope of the first few layers. For the input layer, we prepend a special `<PST>` token. For intermediate layers, we replace the embedding of the `<PST>` token with the sentence embedding decoded from the last token. Finally, we choose a sentence embedding from an intermediate layer as the output (early-exit).

### 4.2 Token Prepending
**4.2.1 Initial Token Prepending:** We prepend a custom placeholder token `<PST>`, which is not in the LLM’s vocabulary, to the input text before the first Transformer layer. We randomly initialize its parameters to maintain the input sequence length.

**4.2.2 Intermediate Token Prepending:** In prepending-enhanced layers, we prepend the sentence embedding token `<SET>` to replace `<PST>` as input to the subsequent layer. This refines the sentence embedding so that subsequent tokens can better capture the sentence’s semantics.

**4.2.3 Layer Scope for Token Prepending:** After several layers, all tokens are contextualized and perceive the complete semantic meaning. Therefore, we stop intermediate token prepending in later layers and directly feed the hidden states into standard Transformer layers.

### 4.3 Early-Exit from Intermediate Layers
Since the last layer of LLMs is primarily used for prediction and contains weaker semantic information, we use embeddings from intermediate layers instead. This improves semantic quality and allows us to obtain sentence embeddings more quickly.

---

## 5. Experiments

### 5.1 Datasets and Experimental Settings
We evaluate on seven STS datasets (STS 2012-2016, STS-B, and SICK-R) using Spearman correlation as the evaluation metric. We use the STS-B development set to determine hyperparameters. 

### 5.2 Baselines
We combine our method with baselines like BERT avg, ST5-Enc avg, LLaMA2 avg, LLaMA2 echo, BERT prompt, PromptEOL, MetaEOL, Pretended CoT, and Knowledge.

### 5.3 Main Results
Our method consistently outperforms all baselines. On LLaMA2-7B, our model shows improvement in 26 out of 28 cases across prompt-based methods. It achieves the most significant improvement with PromptEOL, enhancing performance by 7.16. Furthermore, the inference time of prompt-based methods with TP is within 1.04 times of the original, adding negligible overhead compared to Echo or MetaEOL.

### 5.4 Evaluation of Different Backbones
We evaluate on LLaMA2-7B/13B, Qwen2-7B, LLaMA3-8B, and Gemma2-9B. TP adapts effectively to a range of LLMs, delivering performance gains across different backbones (e.g., 2.17 points improvement on Qwen2-7B).

### 5.5 Analysis of `<PST>` Token
* **Position:** The optimal position of `<PST>` typically places it close to the text (e.g., after the colon). 
* **Ablation:** Retaining `<PST>` after intermediate TP is crucial, as ablating it significantly decreases performance. 
* **Initialization:** The method remains robust regardless of how `<PST>` is initialized (all 0, all 1, Gaussian, etc.).

### 5.6 Analysis of Layer Scope for TP
Performance is sub-optimal if TP doesn't begin at the second layer to replace the randomly initialized `<PST>`. Halting TP after the 7th or 8th layer yields the best performance across most models (LLaMA2, Qwen2, Gemma2).

### 5.7 Influence of Exit Layers
Employing the output of the model’s last layer is consistently suboptimal for STS tasks. The optimal intermediate exit layer shifts depending on the prompt used (e.g., sixth-to-last for Pretended CoT, second-to-last for Knowledge Enhancement).

### 5.8 Transfer Learning Tasks
Evaluated on SentEval tasks (MR, CR, SUBJ, MPQA, SST-2, TREC, MRPC), TP consistently outperforms baselines in 20 out of 21 cases. Transfer tasks benefit from ending token prepending at deeper layers (between 14 and 21).

### 5.9 Evaluation of Capturing Dependencies
We computed the Spearman correlation between the pivot (last) token and remaining tokens. LLaMA2-7B+TP achieved a higher score (25.11) than the vanilla model (23.97), proving TP enhances the ability to capture backward dependencies.

---

## 6. Conclusion
Token Prepending is a plug-and-play, training-free approach for deriving high-quality sentence embeddings from autoregressive LLMs. By intervening in the inputs to Transformer layers, TP enhances backward dependency capturing. It adds negligible inference cost, integrates seamlessly with prompt-based methods, and achieves outstanding performance on STS and transfer learning tasks. The optimal stopping point is typically around the 7th or 8th layer for 7B models.

### Limitations
TP requires tuning two hyperparameters (end layer for intermediate TP and exit layer) which vary based on the model, dataset, and prompt, potentially increasing adaptation costs.

### Acknowledgments
Supported by the National Natural Science Foundation of China and Collaborative Innovation Center of Novel Software Technology and Industrialization.

---

## Appendix A
* **A.1 Bidirectional Attention:** Enabling true bidirectional attention (removing causal masks) leads to a substantial decrease in performance (from 77.54 to 43.70) due to the inductive bias of autoregressive LLMs.
* **A.2 Multi-Task Evaluation:** TP was evaluated on 12 classification datasets, 3 pair classification, 4 reranking, 11 clustering, 1 summarization, and 1 additional STS dataset from the MTEB benchmark. TP showed improvement in 40 out of 44 cases.
* **A.3 More Prompt Baselines:** TP significantly improves custom prompts (Prompt A & B) that do not use prior knowledge by 10.64 and 9.26 points.
* **A.4 Number of `<PST>` tokens:** Using two `<PST>` tokens slightly improves performance, but using three or more leads to a decline.
* **A.5 Masking `<PST>`:** Masking the token in the first layer slightly lowers performance, showing its importance as a length placeholder.
* **A.6 Resuming TP:** Pausing and resuming TP at deeper layers (e.g., layer 21) yields only a slight performance improvement.
* **A.7 TP without Prompt:** Even without a text prompt, applying TP still provides relative improvements in semantic representations.
```

Hy vọng bản tổng hợp này đã cung cấp cho bạn một cái nhìn chi tiết và toàn vẹn nhất về nghiên cứu từ file dữ liệu! Hãy cho tôi biết nếu bạn muốn đi sâu hơn vào bất kỳ khía cạnh nào của phương pháp này.