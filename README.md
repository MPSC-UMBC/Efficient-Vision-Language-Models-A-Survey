# Efficient-Vision-Language-Models
[2025] Efficient Vision Language Models: A Survey  

> **[Paper_Link](http://arxiv.org/abs/2312.03863)**[ [arXiv]](http://arxiv.org/abs/2312.03863) (Paper Versions Listed)

> *Gaurav Shinde*, *Anuradha Ravi*, *Emon Dey*, *Milind Rampure*, *Nirmalya Roy*

> *University of Maryland, Baltimore County (UMBC)*

🤝 Support & Collaboration
We welcome feedback and contributions to improve this survey and repository. The repository will be actively maintained with emerging research. Feel free to reach out via email with any suggestions.

## 📌 Abstract
Vision-language models (VLMs) integrate visual and textual information, enabling a wide range of applications such as image captioning and visual question answering, making them crucial for modern AI systems. However, their high computational demands pose challenges for real-time applications. This has led to a growing focus on developing efficient visionlanguage models. In this survey, we review key techniques
for optimizing VLMs on edge and resource-constrained devices. We also explore compact VLM architectures, frameworks and provide detailed insights into the performancelatency trade-offs of efficient VLMs. Our objective is to foster deeper research in this area.

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
- [🔢 Efficient Finetuning](#Efficient-Finetuning)
  - [Parameter Efficient](#Parameter-Efficient)
    - [Low-Rank Adapters](#Low-Rank-Adapters)
    - [Prompt Tuning](#Prompt-Tuning)
    - [Adapter-based Methods](#Adapter-based-Methods)
    - [Prefaix Tuning](#Prefix-Tuning)
  - [Memory Efficient](#Memory-Efficient)
- [🧑‍💻 Runtime Optimization](#Runtime-Optimization)
    - [Token Reduction](#Token-Reduction)
    - [Test-Time Adaption](#Test-Time-Adaption)
      - [Test-Time Augmentation](#Test-Time-Augmentation)
      - [Test-Time Prompt Tuning](#Test-Time-Prompt-Tuning)
- [🧑‍💻 Privacy-Preserving Distributed VLM](#Privacy-Preserving-Distributed-VLM)


## 🤖 Pre-deployment Techniques
### Quantization
###### Post-Training Quantization
- MBQ-Modality-Balanced Quantization for Large Vision-Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2412.19509)] [[Code](https://github.com/thu-nics/MBQ)]
- Q-VLM- Post-training Quantization for Large Vision Language Models, <ins>NeurIPS, 2024</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cffbaf4f47546ece96bb42c0edda40ee-Abstract-Conference.html)] [[Code](https://github.com/changyuanwang17/qvlm?tab=readme-ov-file)]
- P4Q- Learning to Prompt for Quantization in Visual-language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2409.17634)]
- BiLLM- Pushing the Limit of Post-Training Quantization for LLMs, <ins>ICML, 2024</ins> [[Paper](https://dl.acm.org/doi/10.5555/3692070.3692876)] [[Code](https://github.com/Aaronhuang-778/BiLLM)]
- NoisyQuant- Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers, <ins>IEEE/CVF, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/10203639)] [[Code](https://github.com/kriskrisliu/NoisyQuant?tab=readme-ov-file)]
- LRQuant-Learnable and Robust Post-Training Quantization for Large Language Models, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.acl-long.122/)] [[Code](https://github.com/zjq0455/RLQ)]
- PTQ4ViT- Post-Training Quantization for Vision Transformers with Twin Uniform Quantization, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2111.12293)] [[Code](https://github.com/hahnyuan/PTQ4ViT?tab=readme-ov-file)]
- PTQ4SAM- Post-Training Quantization for Segment Anything, <ins>IEEE/CVF, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/10658486)] [[Code](https://github.com/chengtao-lv/PTQ4SAM)]
###### Quantization-Aware Training
- Boost Vision Transformer with GPU-Friendly Sparsity and Quantization, <ins>IEEE/CVF, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/10203700)]
- Advancing Multimodal Large Language Models with Quantization-aware Scale Learning for Efficient Adaptation, <ins>ACM, 2024</ins> [[Paper](https://dl.acm.org/doi/10.1145/3664647.3680838)] [[Code](https://github.com/xjjxmu/QSLAW?tab=readme-ov-file)]
- Q-ViT- Fully Differentiable Quantization for Vision Transformer, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2201.07703)]
- LLM-QAT-Data-Free Quantization Aware Training for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2305.17888)] [[Code](https://github.com/facebookresearch/LLM-QAT?tab=readme-ov-file)]
- EfficientQAT- Efficient Quantization-Aware Training for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.11062)] [[Code](https://github.com/OpenGVLab/EfficientQAT?tab=readme-ov-file)]
### Low-rank Approximation
- Low-Rank Few-Shot Adaptation of Vision-Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.18541)] [[Code](https://github.com/MaxZanella/CLIP-LoRA?tab=readme-ov-file)]
- Advancing Vision-Language Models with Adapter Ensemble Strategies, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-emnlp.921/)]
### Pruning
###### Structured
- SmartTrim- Adaptive Tokens and Attention Pruning for Efficient VLMs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2305.15033)]
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

## 🔢 Efficient Finetuning
### Parameter Efficient
###### Low-Rank Adapters
###### Prompt Tuning
###### Adapter-based Methods
###### Prefix Tuning
### Memory Efficient
- NeurIPS-2023-make-pre-trained-model-reversible-from-parameter-to-memory-efficient-fine-tuning-Paper-Conference, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.00888)]
- Mercea_Time-_Memory-_and_Parameter-Efficient_Visual_Adaptation_CVPR_2024_paper, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13064)]
- M2IST, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.07625)] [[Code](https://huggingface.co/datasets/math-ai/AutoMathText)]
- SLIMFIT- Memory-Efficient Fine-Tuning of Transformer-based Models using Training Dynamics, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2312.15685)] [[Code](https://github.com/hkust-nlp/deita)]

## 🧑‍💻 Runtime Optimization
### Token Reduction
### Test-Time Adaption
###### Test-Time Augmentation
- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.15627)]
- CoLLiE: Collaborative Training of Large Language Models in an Efficient Way, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2312.00407)] [[Code](https://github.com/OpenLMLab/collie)]
###### Test-Time Prompt Tuning

## 🧑‍💻 Privacy-Preserving Distributed VLM

 <!-- [^1]: This table was updated Dec 2023. This table will require updates as cool new frameworks are being released frequently and current frameworks continue to mature at an accelerated rate. So please feel free to suggest any important distinguishing features or popular new frameworks-->
