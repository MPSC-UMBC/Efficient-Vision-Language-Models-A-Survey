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
### Electroencephalography
### Millimeter Wave Radar
### Audio

 <!-- [^1]: This table was updated Dec 2023. This table will require updates as cool new frameworks are being released frequently and current frameworks continue to mature at an accelerated rate. So please feel free to suggest any important distinguishing features or popular new frameworks-->
