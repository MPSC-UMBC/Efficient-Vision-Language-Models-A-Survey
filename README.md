# Efficient Vision Language Models: A Survey
![image](https://github.com/user-attachments/assets/0711badd-2511-448e-921e-9ab0328ab5fb) ![image](https://github.com/user-attachments/assets/69f88aea-6af0-4676-a979-3662dd8f250c)

> *Paper* - https://arxiv.org/abs/2504.09724

> *Gaurav Shinde*, *Anuradha Ravi*, *Emon Dey*, *Shadman Sakib*, *Milind Rampure*, *Nirmalya Roy*

> *University of Maryland, Baltimore County (UMBC)*


<h4 align="center"> If you like our project, please give us a star ⭐ on GitHub.</h5>
 
<p align="center">
  <a href="https://github.com/MPSC-UMBC/Efficient-Vision-Language-Models-A-Survey/stargazers">
    <img
      src="https://img.shields.io/github/stars/MPSC-UMBC/Efficient-Vision-Language-Models-A-Survey?style=social"
      alt="GitHub stars"
    />
  </a>
</p>

<h3 align="center"> 🚀 Exciting news- Our survey has been accepted at WIREs DMKD (11.7 IF)</h3>

## 🤝 Support & Collaboration
We welcome feedback and contributions to improve this survey and repository. The repository will be actively maintained with emerging research. Feel free to reach out via email (gshinde1@umbc.edu) with any suggestions.

## 📌 Abstract
Vision-language models (VLMs) integrate visual and textual information, enabling a wide range of applications such as image captioning and visual question answering, making them crucial for modern AI systems. However, their high computational demands pose challenges for real-time applications. This has led to a growing focus on developing efficient visionlanguage models. In this survey, we review key techniques
for optimizing VLMs on edge and resource-constrained devices. We also explore compact VLM architectures, frameworks and provide detailed insights into the performancelatency trade-offs of efficient VLMs. Our objective is to foster deeper research in this area.

## 📊 Evaluation of Compact VLMs
![download (4)](https://github.com/user-attachments/assets/59b70ea2-29cb-45d9-8e8f-f8ac8a7d8970)
SmolVLM with 2.2B parameters consistently achieves the highest accuracy across eight benchmarks, followed by the 500M and then the 256M models. Interestingly, Idefics with 80B parameters yields the lowest performance across most benchmarks. In terms of throughput, the 256M model delivers the highest, followed by the 500M, 2.2B, and finally the 80B model. A higher throughput indicates better efficiency. These results highlight a critical insight: achieving optimal performance on edge or pervasive IoT devices requires a careful balance between accuracy and efficiency, tailored to the specific constraints of the deployment hardware, such as memory and GPU availability.


## 📖 Table of Content
- [🚀 Pre-deployment techniques](#Pre-deployment-techniques) 
  - [Quantization](#Quantization)
      - [Post-Training Quantization](#Post-Training-Quantization)
      - [Quantization-Aware Training](#Quantization-Aware-Training)
  - [Low-rank Approximation](#Low-rank-Approximation)
  - [Pruning](#Pruning)
    - [Structured](#Structured)
    - [Unstructured](#Unstructured)
  - [Knowledge Distillation](#Knowledge-Distillation) 
  - [Other Methods](#Other-Methods)
- [🎯 Efficient Finetuning](#Efficient-Finetuning)
  - [Parameter Efficient](#Parameter-Efficient)
    - [Low-Rank Adapters](#Low-Rank-Adapters)
    - [Prompt Tuning](#Prompt-Tuning)
    - [Adapter-based Methods](#Adapter-based-Methods)
    - [Prefix Tuning](#Prefix-Tuning)
  - [Memory Efficient](#Memory-Efficient)
- [⚡ Runtime Optimization](#Runtime-Optimization)
    - [Token Reduction](#Token-Reduction)
    - [Test-Time Adaption](#Test-Time-Adaption)
      - [Test-Time Augmentation](#Test-Time-Augmentation)
      - [Test-Time Prompt Tuning](#Test-Time-Prompt-Tuning)
- [🔒🌐 Privacy-Preserving Distributed VLM](#Privacy-Preserving-Distributed-VLM)


## 🚀 Pre-deployment Techniques
### Quantization
###### Post-Training Quantization
- MBQ-Modality-Balanced Quantization for Large Vision-Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2412.19509)] [[Code](https://github.com/thu-nics/MBQ)]
- Q-VLM- Post-training Quantization for Large Vision Language Models, <ins>NeurIPS, 2024</ins> [[Paper](https://neurips.cc/virtual/2024/poster/94107)] [[Code](https://github.com/changyuanwang17/qvlm?tab=readme-ov-file)]
- P4Q- Learning to Prompt for Quantization in Visual-language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2409.17634)]
- BiLLM- Pushing the Limit of Post-Training Quantization for LLMs, <ins>ICML, 2024</ins> [[Paper](https://dl.acm.org/doi/10.5555/3692070.3692876)] [[Code](https://github.com/Aaronhuang-778/BiLLM)]
- NoisyQuant- Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers, <ins>CVPR, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/10203639)] [[Code](https://github.com/kriskrisliu/NoisyQuant?tab=readme-ov-file)]
- LRQuant-Learnable and Robust Post-Training Quantization for Large Language Models, <ins>ACL, 2024</ins> [[Paper](https://aclanthology.org/2024.acl-long.122/)] [[Code](https://github.com/zjq0455/RLQ)]
- PTQ4ViT- Post-Training Quantization for Vision Transformers with Twin Uniform Quantization, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2111.12293)] [[Code](https://github.com/hahnyuan/PTQ4ViT?tab=readme-ov-file)]
- PTQ4SAM- Post-Training Quantization for Segment Anything, <ins>CVPR, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10658486)] [[Code](https://github.com/chengtao-lv/PTQ4SAM)]
- SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization<ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2411.10958)] [[Code](https://github.com/thu-ml/SageAttention)]
- SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration <ins>arXiv, 2024</ins> [[Paper](https://doi.org/10.48550/arXiv.2410.02367)] [[Code](https://github.com/thu-ml/SageAttention)]
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning <ins>arXiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.08691)] [[Code](https://github.com/Dao-AILab/flash-attention)]
- Post-Training Quantization for Vision Transformer,<ins>arXiv, 2021</ins> [[Paper](https://doi.org/10.48550/arXiv.2106.14156)] 
- FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer,<ins>arXiv, 2021</ins> [[Paper](https://doi.org/10.48550/arXiv.2111.13824)] [[Code](https://github.com/megvii-research/FQ-ViT)]
- SmoothQuant: accurate and efficient post-training quantization for large language models,<ins>arXiv, 2022</ins> [[Paper](https://doi.org/10.48550/arXiv.2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]
###### Quantization-Aware Training
- Boost Vision Transformer with GPU-Friendly Sparsity and Quantization, <ins>CVPR, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/10203700)]
- Advancing Multimodal Large Language Models with Quantization-aware Scale Learning for Efficient Adaptation, <ins>ACM, 2024</ins> [[Paper](https://dl.acm.org/doi/10.1145/3664647.3680838)] [[Code](https://github.com/xjjxmu/QSLAW?tab=readme-ov-file)]
- Q-ViT- Fully Differentiable Quantization for Vision Transformer, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2201.07703)]
- LLM-QAT-Data-Free Quantization Aware Training for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2305.17888)] [[Code](https://github.com/facebookresearch/LLM-QAT?tab=readme-ov-file)]
- EfficientQAT- Efficient Quantization-Aware Training for Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.11062)] [[Code](https://github.com/OpenGVLab/EfficientQAT?tab=readme-ov-file)]
- Quantization Variation: A New Perspective on Training Transformers with Low-Bit Precision, <ins>arXiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2307.00331)] [[Code](https://github.com/HuangOwen/Quantization-Variation)]
- PackQViT: faster sub-8-bit vision transformers via full and packed quantization on the mobile, <ins>ACM, 2023</ins> [[Paper](https://dl.acm.org/doi/10.5555/3666122.3666518)] [[Code](https://github.com/PeiyanFlying/PackQViT)]
- Fixed-point Quantization for Vision Transformer, <ins>IEEE, 2021</ins> [[Paper](https://ieeexplore.ieee.org/document/9728246)]
- QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models<ins>arXiv, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2309.14717)] [[Code](https://github.com/yuhuixu1993/qa-lora)]
- Low-Rank Quantization-Aware Training for LLMs <ins>arXiv, 2024</ins> [[Paper](https://doi.org/10.48550/arXiv.2406.06385)] [[Code](https://github.com/qualcomm-ai-research/LR-QAT)]

### Low-rank Approximation
- SeTAR: Out-of-Distribution Detection with Selective Low-Rank Approximation, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.12629)] [[Code](https://github.com/X1AOX1A/SeTAR)]
- PELA: Learning Parameter-Efficient Models with Low-Rank Approximation, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.10700)] [[Code](https://github.com/guoyang9/PELA)]
- Low-Rank Approximation for Sparse Attention in Multi-Modal LLMs, <ins>CVPR, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10655939)]
### Pruning
###### Structured
- SmartTrim- Adaptive Tokens and Attention Pruning for Efficient VLMs, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.15033)]
- OSSCAR- One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization, <ins>ICML, 2024</ins> [[Paper](https://dl.acm.org/doi/10.5555/3692070.3693510)] [[Code](https://github.com/mazumder-lab/OSSCAR)]
- MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric, <ins>CVPR, 2024</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_MoPE-CLIP_Structured_Pruning_for_Efficient_Vision-Language_Models_with_Module-wise_Pruning_CVPR_2024_paper.html)]
- Isomorphic Pruning for Vision Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.04616)] [[Code](https://github.com/VainF/Isomorphic-Pruning?tab=readme-ov-file)]
###### Unstructured
- Rethinking Pruning for Vision-Language Models: Strategies for Effective Sparsity and Performance Restoration, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.02424)] [[Code](https://github.com/Shwai-He/VLM-Compression?tab=readme-ov-file)]
- MULTIFLOW- Shifting Towards Task-Agnostic Vision-Language Pruning, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2404.05621)] [[Code](https://github.com/FarinaMatteo/multiflow)]
- ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2310.02998)] [[Code](https://github.com/ylsung/ECoFLaP)]
### Knowledge Distillation
- VLDadaptor_Domain_Adaptive_Object_Detection_With_Vision-Language_Model_Distillation, <ins>IEEE TMM, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10669066)] [[Code](https://github.com/GingerCohle/VLDadaptor?tab=readme-ov-file)]
- Building Vision-Language Models on Solid Foundations with Masked Distillation, <ins>CVPR, 2024</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Sameni_Building_Vision-Language_Models_on_Solid_Foundations_with_Masked_Distillation_CVPR_2024_paper.html)]
- PromptKD: Unsupervised Prompt Distillation for Vision-Language Models, <ins>CVPR, 2024</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html)] [[Code](https://github.com/zhengli97/PromptKD?tab=readme-ov-file)]
- KD-VLP: Improving End-to-End Vision-and-Language Pretraining with Object Knowledge Distillation, <ins>arXiv, 2021</ins> [[Paper](https://arxiv.org/abs/2109.10504)]
- Compressing Visual-linguistic Model via Knowledge Distillation, <ins>ICCV, 2021</ins> [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Compressing_Visual-Linguistic_Model_via_Knowledge_Distillation_ICCV_2021_paper.html)]
### Other Methods
- Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning, <ins>CVPR, 2017</ins> [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Lu_Knowing_When_to_CVPR_2017_paper.html)]
- Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2403.11549)] [[Code](https://github.com/JiazuoYu/MoE-Adapters4CL?tab=readme-ov-file)]
- Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.10237)] [[Code](https://github.com/jiangsongtao/Med-MoE?tab=readme-ov-file)]
- Scaling Vision-Language Models with Sparse Mixture of Experts, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2303.07226)]

## 🎯 Efficient Finetuning
### Parameter Efficient
###### Low-Rank Adapters
- Low-Rank Few-Shot Adaptation of Vision-Language Models , <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2405.18541)] [[Code](https://github.com/MaxZanella/CLIP-LoRA?tab=readme-ov-file)]
- Advancing Vision-Language Models with Adapter Ensemble Strategies , <ins>EMNLP, 2024</ins> [[Paper](https://aclanthology.org/2024.findings-emnlp.921/)]
- Enhancing Adversarial Robustness of Vision-Language Models through Low-Rank Adaptation, <ins>arXiv, 2024</ins> [[Paper](https://doi.org/10.48550/arXiv.2404.13425)]
- Complementary Subspace Low-Rank Adaptation of Vision-Language Models for Few-Shot Classification, <ins>arXiv, 2025</ins> [[Paper](https://doi.org/10.48550/arXiv.2501.15040)]
- One Head Eight Arms: Block Matrix based Low Rank Adaptation for CLIP-based Few-Shot Learning, <ins>arXiv, 2025</ins> [[Paper](https://doi.org/10.48550/arXiv.2501.16720)]
- Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA, <ins>arXiv, 2024</ins> [[Paper](https://doi.org/10.48550/arXiv.2412.01004)] [[Code](https://github.com/jeff024/DyRA)]
- LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models, <ins>arXiv, 2025</ins> [[Paper](https://doi.org/10.48550/arXiv.2502.02069)]
- Serial Low-rank Adaptation of Vision Transformer, <ins>arXiv, 2025</ins> [[Paper](https://doi.org/10.48550/arXiv.2503.17750)]

###### Prompt Tuning
- Visual-Language Prompt Tuning with Knowledge-guided Context Optimization, <ins>CVPR, 2023</ins> [[Paper](https://arxiv.org/abs/2303.13283)] [[Code](https://github.com/htyao89/KgCoOp?tab=readme-ov-file)]
- Dual Modality Prompt Tuning for Vision-Language Pre-Trained Model,<ins>IEEE TMM, 2023</ins> [[Paper](https://dl.acm.org/doi/10.1109/TMM.2023.3291588)] [[Code](https://github.com/fanrena/dpt)]
- Distribution-Aware Prompt Tuning for Vision-Language Models,<ins>ICCV, 2023</ins> [[Paper](https://www.computer.org/csdl/proceedings-article/iccv/2023/071800v1947/1TJdQhTQazK)] [[Code](https://github.com/mlvlab/DAPT)]
  
###### Adapter-based Methods
- MMA: Multi-Modal Adapter for Vision-Language Models,<ins>CVPR, 2024</ins> [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.html)] [[Code](https://github.com/ZjjConan/VLM-MultiModalAdapter)]
- VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks,<ins>CVPR, 2022</ins> [[Paper](https://arxiv.org/abs/2112.06825)] [[Code](https://github.com/ylsung/VL_adapter)]
- Meta-Adapter: An Online Few-shot Learner for Vision-Language Mode,<ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2311.03774)][[Code](https://github.com/ArsenalCheng/Meta-Adapter)]
  
###### Prefix Tuning
- Open-Ended Medical Visual Question Answering Through Prefix Tuning of Language Models,<ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2303.05977)] [[Code](https://github.com/tjvsonsbeek/open-ended-medical-vqa?tab=readme-ov-file)]
- User-Aware Prefix-Tuning is a Good Learner for Personalized Image Captioning,<ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2312.04793)] 
- Context-aware Visual Storytelling with Visual Prefix Tuning and Contrastive Learning,<ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2408.06259)]

### Memory Efficient
- Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2306.00477)] [[Code](https://github.com/BaohaoLiao/mefts)]
- Time-, Memory- and Parameter-Efficient Visual Adaptation, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2402.02887)]
- M2IST: Multi-Modal Interactive Side-Tuning for Efficient Referring Expression Comprehension, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2407.01131)] [[Code](https://github.com/xuyang-liu16/M2IST)]
- SLIMFIT- Memory-Efficient Fine-Tuning of Transformer-based Models using Training Dynamics, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2305.18513)] [[Code](https://github.com/arashardakani/SlimFit)]

## ⚡ Runtime Optimization

### Token Reduction
- Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.07408)]
- PuMer: Pruning and Merging Tokens for Efficient Vision Language Models, <ins>ACL 2023</ins> [[Paper](https://aclanthology.org/2023.acl-long.721/)] [[Code](https://github.com/csarron/PuMer)]

### Test-Time Adaption
- Frustratingly Easy Test-Time Adaptation of Vision-Language Models, <ins>NeurIPS, 2024</ins> [[Paper](https://neurips.cc/virtual/2024/poster/94270)] [[Code](https://github.com/FarinaMatteo/zero)]
- Efficient Test-Time Adaptation of Vision-Language Models, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2403.18293)] [[Code](https://github.com/kdiAAA/TDA)]
- SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models, <ins>NeurIPS, 2023</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html#:~:text=The%20proposed%20SwapPrompt%20can%20be,ImageNet%20and%20nine%20other%20datasets.)]
- Online Gaussian Test-Time Adaptation of Vision-Language Models, <ins>arXiv 2025</ins> [[Paper](https://arxiv.org/abs/2501.04352)] [[Code](https://github.com/cfuchs2023/OGA)]

###### Test-Time Augmentation
- TextAug: Test time Text Augmentation for Multimodal Person Re-identification, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2312.01605)]
- On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning?, <ins>CVPR, 2024</ins> [[Paper](https://arxiv.org/abs/2405.02266)] [[Code](https://github.com/MaxZanella/MTA)]

###### Test-Time Prompt Tuning
- Efficient Test-Time Prompt Tuning for Vision-Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2408.05775)]
- C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion, <ins>ICLR, 2024</ins> [[Paper](https://openreview.net/forum?id=jzzEHTBFOT)] [[Code](https://github.com/hee-suk-yoon/C-TPT)]
- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models, <ins>NeurIPS, 2022</ins> [[Paper](https://arxiv.org/abs/2209.07511)] [[Code](https://github.com/azshue/TPT)]

## 🔒🌐 Privacy-Preserving Distributed VLM

- pFedPrompt: Learning Personalized Prompt for Vision-Language Models in Federated Learning, <ins>ACM WWW, 2023</ins> [[Paper](https://dl.acm.org/doi/10.1145/3543507.3583518)]
- Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method, <ins>NeurIPS, 2024</ins> [[Paper](https://nips.cc/virtual/2024/poster/94723)] [[Code](https://github.com/PanBikang/PromptFolio)]
- Efficient Adapting for Vision-language Foundation Model in Edge Computing Based on Personalized and Multi-Granularity Federated Learning, <ins>IEEE INFOCOM, 2024</ins> [[Paper](https://ieeexplore.ieee.org/document/10620835)]
- Fair Federated Learning with Biased Vision-Language Models, <ins>ACL 2024</ins> [[Paper](https://aclanthology.org/2024.findings-acl.595/)]
- Lightweight Unsupervised Federated Learning with Pretrained Vision Language Model, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2404.11046)]

## 📖 Citation

If you find this survey valuable, please cite:

```bibtex
@misc{shinde2025surveyefficientvisionlanguagemodels,
      title={A Survey on Efficient Vision-Language Models}, 
      author={Gaurav Shinde and Anuradha Ravi and Emon Dey and Shadman Sakib and Milind Rampure and Nirmalya Roy},
      year={2025},
      eprint={2504.09724},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.09724}, 
}
