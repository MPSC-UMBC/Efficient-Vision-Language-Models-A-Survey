# VLM_Survey_Paper
[2025] Efficient Vision Language Models: A Survey  

> **[Paper_Link](http://arxiv.org/abs/2312.03863)**[ [arXiv]](http://arxiv.org/abs/2312.03863) (Paper Versions Listed)

> *Authors*

> *University*

## 📌 What is This Survey About?

Paper Description

## 📖 Table of Content
- [🤖 Pre-deployment techniques](#Pre-deployment-techniques) 
  - [Quantization](#Quantization)
      - [Post-Training Quantization](#Post-Training-Quantization)
      - [Quantization-Aware Training](#Quantization-Aware-Training)
  - [Low-rank Approximation](#Low-rank-Approximation)
  - [Pruning](#Pruning)
    - [Structured](#Structured)
    - [Unstructured](#Unstructured)
  - [Knowledge Distillation](#Knowledge-Distillation) 
  - [Other Methods](#Other-Methods)
- [🔢 Post-deployment Techniques](#Post-deployment-Techniques)
  - [Efficient Finetuning](#Efficient-Finetuning)
    - [Parameter Efficient](#Parameter-Efficient)
      - [Low-Rank Adapters](#Low-Rank-Adapters)
      - [Prompt Tuning](#Prompt-Tuning)
      - [Adapter-based Methods](#Adapter-based-Methods)
      - [Mapping-based Methods](#Mapping-based-Methods)
    - [Memory Efficient](#Memory-Efficient)
- [🧑‍💻 Runtime Optimization](#Runtime-Optimization)
    - [Scheduling](#Scheduling)
    - [Batching](#Batching)
    - [Hardware Optimization](#Hardware-Optimization)
- [🧑‍💻 Augmenting modalities with VLMs](#Augmenting-modalities-with-VLMs)
    - [Electroencephalography](#Electroencephalography)
    - [Millimeter Wave Radar](#Millimeter-Wave-Radar)
    - [Audio](#Audio)


## 🤖 Pre-deployment Techniques
### Quantization
###### Post-Training Quantization
- MBQ-Modality-Balanced Quantization for Large Vision-Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- Q-VLM- Post-training Quantization for Large Vision Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- P4Q- Learning to Prompt for Quantization in Visual-language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- BiLLM- Pushing the Limit of Post-Training Quantization for LLMs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.11295)]
- NoisyQuant- Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
- LRQuant-Learnable and Robust Post-Training Quantization for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.13304)] [[Code](https://github.com/jerry-chee/QuIP)]
- PTQ4ViT- Post-Training Quantization for Vision Transformers with Twin Uniform Quantization, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- PTQ4SAM- Post-Training Quantization for Segment Anything, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.02272)] [[Code](https://github.com/xvyaward/owq)]
###### Quantization-Aware Training
- Boost Vision Transformer with GPU-Friendly Sparsity and Quantization, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- Advancing Multimodal Large Language Models with Quantization-aware Scale Learning for Efficient Adaptation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- Q-ViT- Fully Differentiable Quantization for Vision Transformer, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- LLM-QAT-Data-Free Quantization Aware Training for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.11295)]
- EfficientQAT- Efficient Quantization-Aware Training for Large Language Models, <ins>ICLR, 2023</ins> [[Paper](https://openreview.net/forum?id=tcbBPnfwxS)] [[Code](https://github.com/IST-DASLab/gptq)]
### Low-rank Approximation
- SeTAR- Out-of-Distribution Detection with selective low-rank approximation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- PELA- Learning Parameter-Efficient Models with Low-Rank Approximation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- Low-Rank Approximation for Sparse Attention in Multi-Modal LLMs, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
### Pruning
###### Structured
- SmartTrim- Adaptive Tokens and Attention Pruning for Efficient VLMs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- OSSCAR- One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- Lin_MoPE-CLIP_Structured_Pruning_for_Efficient_Vision-Language_Models_with_Module-wise_Pruning_CVPR_2024_paper, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- Isomorphic Pruning for Vision Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
###### Unstructured
- Rethinking Pruning for Vision-Language Models- Strategies for effective sparsity and performance restoration, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.17849)] 
- MULTIFLOW- Shifting Towards Task-Agnostic Vision-Language Pruning, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.01241)] 
- ECOFLAP- EFFICIENT COARSE-TO-FINE LAYER-WISE PRUNING FOR VISION-LANGUAGE MODELS, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
### Knowledge Distillation
- VLDadaptor_Domain_Adaptive_Object_Detection_With_Vision-Language_Model_Distillation, <ins>NeurIPS, 2023</ins> [[Paper](https://openreview.net/forum?id=bqGK5PyI6-N)] [[Code](https://github.com/rabeehk/compacter)]
- Sameni_Building_Vision-Language_Models_on_Solid_Foundations_with_Masked_Distillation_CVPR_2024_paper, <ins>NeurIPS, 2022</ins> [[Paper](https://openreview.net/forum?id=rBCvMG-JsPd)] [[Code](https://github.com/r-three/t-few)]
- Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper, <ins>AutoML, 2022</ins> [[Paper](https://openreview.net/forum?id=BCGNf-prLg5)]
- KD-VLP, <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.388/)] [[Code](https://github.com/microsoft/AdaMix)]
- Fang_Compressing_Visual-Linguistic_Model_via_Knowledge_Distillation_ICCV_2021_paper, <ins>EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.findings-emnlp.160/)] [[Code](https://github.com/Shwai-He/SparseAdapter)]
### Other Methods
- Lu_Knowing_When_to_CVPR_2017_paper, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/html/2408.11049v1)]
- Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.00242)]
- Med-MoE- Mixture of Domain-Specific Experts for Lightweight Medical VLMs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.16710)]
- Scaling Vision-Language Models with Sparse Mixture of Experts, <ins>arXiv, 2024</ins> [[Paper](https://github.com/Infini-AI-Lab/TriForce)]

## 🔢 Post-deployment Techniques
### Efficient Finetuning
#### Parameter Efficient
###### Low-Rank Adapters
###### Prompt Tuning
###### Adapter-based Methods
###### Mapping-based Methods
#### Memory Efficient
- NeurIPS-2023-make-pre-trained-model-reversible-from-parameter-to-memory-efficient-fine-tuning-Paper-Conference, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.00888)]
- Mercea_Time-_Memory-_and_Parameter-Efficient_Visual_Adaptation_CVPR_2024_paper, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13064)]
- M2IST, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.07625)] [[Code](https://huggingface.co/datasets/math-ai/AutoMathText)]
- SLIMFIT- Memory-Efficient Fine-Tuning of Transformer-based Models using Training Dynamics, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2312.15685)] [[Code](https://github.com/hkust-nlp/deita)]

## 🧑‍💻 Runtime Optimization
### Scheduling
### Batching
### Hardware Optimization
- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.15627)]
- CoLLiE: Collaborative Training of Large Language Models in an Efficient Way, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2312.00407)] [[Code](https://github.com/OpenLMLab/collie)]

## 🧑‍💻 Augmenting modalities with VLMs
### Scheduling
### Batching
### Hardware Optimization

## System-Level Serving Efficiency Optimization
##### Serving System Design
- LUT TENSOR CORE: Lookup Table Enables Efficient Low-Bit LLM Inference Acceleration, <ins>arXiv, 2024</ins> [[Paper](https://paperswithcode.com/paper/lut-tensor-core-lookup-table-enables)]
- TurboTransformers: an efficient GPU serving system for transformer models, <ins>PPoPP, 2021</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3437801.3441578)]
- Orca: A Distributed Serving System for Transformer-Based Generative Models, <ins>OSDI, 2022</ins> [[Paper](https://www.usenix.org/conference/osdi22/presentation/yu)]
- FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU, <ins>ICML, 2023</ins> [[Paper](https://arxiv.org/abs/2303.06865)] [[Code](https://github.com/FMInference/FlexGen)]
- Efficiently Scaling Transformer Inference, <ins>MLSys, 2023</ins> [[Paper](https://proceedings.mlsys.org/paper_files/paper/2023/file/523f87e9d08e6071a3bbd150e6da40fb-Paper-mlsys2023.pdf)]
- DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale, <ins>SC, 2022</ins> [[Paper](https://dl.acm.org/doi/abs/10.5555/3571885.3571946)]
- Efficient Memory Management for Large Language Model Serving with PagedAttention, <ins>SOSP, 2023</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3600006.3613165)] [[Code](https://github.com/vllm-project/vllm)]
- S-LoRA: Serving Thousands of Concurrent LoRA Adapters, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2311.03285)] [[Code](https://github.com/S-LoRA/S-LoRA)]
- Petals: Collaborative Inference and Fine-tuning of Large Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2209.01188)] 
- SpotServe: Serving Generative Large Language Models on Preemptible Instances, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.15566)]

##### Serving Performance Optimization
- KV-Runahead: Scalable Causal LLM Inference by Parallel Key-Value Cache Generation, <ins>arXiv, ICML</ins> [[Paper](https://arxiv.org/abs/2405.05329)]
- CacheGen: KV Cache Compression and Streaming for Fast Language Model Serving, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2310.07240)]
- Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding, <ins>TMLR, 2024</ins> [[Paper](https://openreview.net/forum?id=yUmJ483OB0)]
- Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.10285)]
- S3: Increasing GPU Utilization during Generative Inference for Higher Throughput, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.06000)]
- Fast Distributed Inference Serving for Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.05920)]
- Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.13144)]
- SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.16369)]
- FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.05176)]
- Prompt Cache: Modular Attention Reuse for Low-Latency Inference, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.04934)]
- Fairness in Serving Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2401.00588)]

#### Algorithm-Hardware Co-Design
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.08608)] 
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2205.14135)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.08691)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- Flash-Decoding for Long-Context Inference, <ins>Blog, 2023</ins> [[Blog](https://pytorch.org/blog/flash-decoding/)]
- FlashDecoding++: Faster Large Language Model Inference on GPUs, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.01282)]
- PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.12456)] [[Code](https://github.com/SJTU-IPADS/PowerInfer)]
- LLM in a flash: Efficient Large Language Model Inference with Limited Memory, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.11514)]
- Chiplet Cloud: Building AI Supercomputers for Serving Large Generative Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.02666)]
- EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2308.14352)]


### LLM Frameworks
<div align="center">

|                                                    | Efficient Training | Efficient Inference | Efficient Fine-Tuning    |
| :-------------------------------------------------------------------- | :------------------: | :---------------------: | :--: |
| DeepSpeed [[Code](https://github.com/microsoft/DeepSpeed)]            | ✅                   | ✅                     | ✅   |
| Megatron [[Code](https://github.com/NVIDIA/Megatron-LM)]              | ✅                   | ✅                     | ✅   |
| ColossalAI [[Code](https://github.com/hpcaitech/ColossalAI)]          | ✅                   | ✅                     | ✅   |
| Nanotron [[Code](https://github.com/huggingface/nanotron)]            | ✅                   | ✅                     | ✅   |
| MegaBlocks [[Code](https://github.com/databricks/megablocks)]         | ✅                   | ✅                     | ✅   |
| FairScale [[Code](https://github.com/facebookresearch/fairscale)]     | ✅                   | ✅                     | ✅   |
| Pax [[Code](https://github.com/google/paxml/)]                        | ✅                   | ✅                     | ✅   |
| Composer [[Code](https://github.com/mosaicml/composer)]               | ✅                   | ✅                     | ✅   |
| OpenLLM [[Code](https://github.com/bentoml/OpenLLM)]                  | ❌                   | ✅                     | ✅   |
| LLM-Foundry [[Code](https://github.com/mosaicml/llm-foundry)]         | ❌                   | ✅                     | ✅   |
| vLLM [[Code](https://github.com/vllm-project/vllm)]                   | ❌                   | ✅                     | ❌   |
| TensorRT-LLM [[Code](https://github.com/NVIDIA/TensorRT-LLM)]         | ❌                   | ✅                     | ❌   |
| TGI [[Code](https://github.com/huggingface/text-generation-inference)]| ❌                   | ✅                     | ❌   |
| RayLLM [[Code](https://github.com/ray-project/ray-llm)]              | ❌                   | ✅                     | ❌   |
| MLC LLM [[Code](https://github.com/mlc-ai/mlc-llm)]                   | ❌                   | ✅                     | ❌   |
| Sax [[Code](https://github.com/google/saxml)]                         | ❌                   | ✅                     | ❌   |
| Mosec [[Code](https://github.com/mosecorg/mosec)]                     | ❌                   | ✅                     | ❌   |

</div>

 <!-- [^1]: This table was updated Dec 2023. This table will require updates as cool new frameworks are being released frequently and current frameworks continue to mature at an accelerated rate. So please feel free to suggest any important distinguishing features or popular new frameworks-->
