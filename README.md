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
      - [Adapter-based Methods]
      - [Mapping-based Methods]
    - [Memory Efficient](#Memory-Efficient)
  - [Prompt Engineering](#Prompt-Engineering)
    - [Few-Shot Prompting](#Few-Shot-Prompting)
      - [Demonstration Organization](#Demonstration-Organization)
        - [Demonstration Selection](#Demonstration-Selection)
        - [Demonstration Ordering](#Demonstration-Ordering)
      - [Template Formatting](#Template-Formatting)
        - [Instruction Generation](#Instruction-Generation)
        - [Multi-Step Reasoning](#Multi-Step-Reasoning)
        - [Parallel Generation](#Parallel-Generation)
    - [Prompt Compression](#Prompt-Compression)
    - [Prompt Generation](#Prompt-Generation)
- [🧑‍💻 System-Level Efficiency Optimization and LLM Frameworks](#-System-Level-Efficiency-Optimization-and-LLM-Frameworks)
    - [System-Level Efficiency Optimization](#System-Level-Efficiency-Optimization)
      - [System-Level Pre-Training Efficiency Optimization](#System-Level-Pre-Training-Efficiency-Optimization)
      - [System-Level Serving Efficiency Optimization](#System-Level-Serving-Efficiency-Optimization)
        - [Serving System Design](#Serving-System-Design)
        - [Serving Performance Optimization](#Serving-Performance-Optimization)
      - [Algorithm-Hardware Co-Design](#Algorithm-Hardware-Co-Design)
    - [LLM Frameworks](#LLM-Frameworks)

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
##### Low-Rank Adapters
##### Prompt Tuning
##### Adapter-based Methods
##### Mapping-based Methods
- MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.06046)]
- DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.10429)]
- Data Selection for Language Models via Importance Resampling, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2302.03169)] [[Code](https://github.com/p-lambda/dsir)]
- NLP From Scratch Without Large-Scale Pretraining: A Simple and Efficient Framework, <ins>ICML, 2022</ins> [[Paper](https://arxiv.org/pdf/2111.04130.pdf)] [[Code](https://github.com/yaoxingcheng/TLM)]
- Span Selection Pre-training for Question Answering, <ins>ACL, 2020</ins> [[Paper](https://arxiv.org/abs/1909.04120)] [[Code](https://github.com/IBM/span-selection-pretraining)]
#### Memory Efficient
- NeurIPS-2023-make-pre-trained-model-reversible-from-parameter-to-memory-efficient-fine-tuning-Paper-Conference, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2406.00888)]
- Mercea_Time-_Memory-_and_Parameter-Efficient_Visual_Adaptation_CVPR_2024_paper, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.13064)]
- M2IST, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.07625)] [[Code](https://huggingface.co/datasets/math-ai/AutoMathText)]
- SLIMFIT- Memory-Efficient Fine-Tuning of Transformer-based Models using Training Dynamics, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2312.15685)] [[Code](https://github.com/hkust-nlp/deita)]
### Prompt Engineering
#### Few-Shot Prompting
##### Demonstration Organization
###### Demonstration Selection  
- Unified Demonstration Retriever for In-Context Learning, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2305.04320)] [[Code](https://arxiv.org/abs/2305.04320)]
- Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning, <ins> NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2301.11916)] [[Code](https://github.com/WANGXinyiLinda/concept-based-demonstration-selection)]
- In-Context Learning with Iterative Demonstration Selection, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2310.09881)]
- Dr.ICL: Demonstration-Retrieved In-context Learning, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2305.14128)]
- Learning to Retrieve In-Context Examples for Large Language Models, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2307.07164)]
- Finding Supporting Examples for In-Context Learning, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2302.13539)]
- Self-Adaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering, <ins>ACL, 2023</ins> [[Paper](https://aclanthology.org/2023.acl-long.79.pdf)] [[Code](https://github.com/Shark-NLP/self-adaptive-ICL)]
- Selective Annotation Makes Language Models Better Few-Shot Learners, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2209.01975)] [[Code](https://github.com/xlang-ai/icl-selective-annotation)]
- What Makes Good In-Context Examples for GPT-3? <ins>DeeLIO, 2022</ins> [[Paper](https://arxiv.org/abs/2101.06804)]
- Learning To Retrieve Prompts for In-Context Learning, <ins>NAACL-HLT, 2022</ins> [[Paper](https://arxiv.org/abs/2112.08633)] [[Code](https://github.com/OhadRubin/EPR)]
- Active Example Selection for In-Context Learning, <ins> EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.622/)] [[Code](https://github.com/chicagohai/active-example-selection)]
- Rethinking the Role of Demonstrations: What makes In-context Learning Work? <ins> EMNLP, 2022</ins> [[Paper](https://aclanthology.org/2022.emnlp-main.759.pdf)] [[Code](https://github.com/Alrope123/rethinking-demonstrations)]
###### Demonstration Ordering
- Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity, <ins>ACL, 2022</ins> [[Paper](https://arxiv.org/abs/2104.08786)]
##### Template Formatting
###### Instruction Generation
- Large Language Models as Optimizers, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.03409)]
- Instruction Induction: From Few Examples to Natural Language Task Descriptions, <ins>ACL, 2023</ins> [[Paper](https://arxiv.org/abs/2205.10782)] [[Code](https://github.com/orhonovich/instruction-induction)]
- Large Language Models Are Human-Level Prompt Engineers, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2211.01910)] [[Code](https://github.com/keirp/automatic_prompt_engineer)]
- TeGit: Generating High-Quality Instruction-Tuning Data with Text-Grounded Task Design, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2309.05447)]
- Self-Instruct: Aligning Language Model with Self Generated Instructions, <ins>ACL, 2023</ins> [[Paper](https://doi.org/10.48550/arXiv.2212.10560)] [[Code](https://github.com/yizhongw/self-instruct)]
###### Multi-Step Reasoning
- ReST-MCTS∗: LLM Self-Training via Process Reward Guided Tree Search, <ins>NeurIPS, 2024</ins> [[Paper](https://arxiv.org/abs/2406.03816)] 
- Training Language Models to Self-Correct via Reinforcement Learning, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2409.12917)] 
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2408.03314)] 
- Learning to Reason with LLMs, <ins>Website, 2024</ins> [[Html](https://openai.com/index/learning-to-reason-with-llms/)] 
- Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.09629)] 
- From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2405.14838)] 
- Automatic Chain of Thought Prompting in Large Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2210.03493)] [[Code](https://github.com/amazon-science/auto-cot)]
- Measuring and Narrowing the Compositionality Gap in Language Models, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2210.03350)] [[Code](https://github.com/ofirpress/self-ask)]
- ReAct: Synergizing Reasoning and Acting in Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2210.03629)] [[Code](https://github.com/ysymyth/ReAct)]
- Least-to-Most Prompting Enables Complex Reasoning in Large Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2205.10625)]
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09687)] [[Code](https://github.com/spcl/graph-of-thoughts)]
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models, <ins>NeurIPS, 2023</ins> [[Paper](https://arxiv.org/abs/2305.10601)] [[Code](https://github.com/princeton-nlp/tree-of-thought-llm)]
- Self-Consistency Improves Chain of Thought Reasoning in Language Models, <ins>ICLR, 2023</ins> [[Paper](https://arxiv.org/abs/2203.11171)]
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.09687)] [[Code](https://github.com/spcl/graph-of-thoughts)]
- Contrastive Chain-of-Thought Prompting, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/pdf/2311.09277.pdf)] [[Code](https://github.com/DAMO-NLP-SG/contrastive-cot)]
- Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2311.04254)]
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, <ins>NeurIPS, 2022</ins> [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)]
###### Parallel Generation
- Better & Faster Large Language Models via Multi-token Prediction, <ins>arXiv, 2023</ins>  [[Paper](https://arxiv.org/abs/2404.19737)] 
- Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding, <ins>arXiv, 2023</ins>  [[Paper](https://arxiv.org/abs/2307.15337)] [[Code](https://github.com/imagination-research/sot)]
#### Prompt Compression
- LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.12968)]
- PCToolkit: A Unified Plug-and-Play Prompt Compression Toolkit of Large Language Models, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2403.17411)]
- Compressed Context Memory For Online Language Model Interaction, <ins>ICLR, 2024</ins> [[Paper](https://arxiv.org/abs/2312.03414)]
- Learning to Compress Prompts with Gist Tokens, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2304.08467)]
- Adapting Language Models to Compress Contexts, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2305.14788)] [[Code](https://github.com/princeton-nlp/AutoCompressors)]
- In-context Autoencoder for Context Compression in a Large Language Model, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2307.06945)] [[Code](https://github.com/getao/icae)]
- LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.06839)] [[Code](https://github.com/microsoft/LLMLingua)]
- Discrete Prompt Compression with Reinforcement Learning, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2308.08758)]
- Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models, <ins>arXiv, 2023</ins> [[Paper](https://arxiv.org/abs/2310.02409)]
#### Prompt Generation  
- TempLM: Distilling Language Models into Template-Based Generators, <ins>arXiv, 2022</ins> [[Paper](https://arxiv.org/abs/2205.11055)] [[Code](https://github.com/Tiiiger/templm)]
- PromptGen: Automatically Generate Prompts using Generative Models, <ins>NAACL Findings, 2022</ins> [[Paper](https://aclanthology.org/2022.findings-naacl.3/)]
- AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts, <ins>EMNLP, 2020</ins> [[Paper](https://aclanthology.org/2020.emnlp-main.346.pdf)] [[Code](https://ucinlp.github.io/autoprompt/)]

## 🧑‍💻 System-Level Efficiency Optimization and LLM Frameworks
### System-Level Efficiency Optimization
#### System-Level Pre-Training Efficiency Optimization  
- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs, <ins>arXiv, 2024</ins> [[Paper](https://arxiv.org/abs/2402.15627)]
- CoLLiE: Collaborative Training of Large Language Models in an Efficient Way, <ins>EMNLP, 2023</ins> [[Paper](https://arxiv.org/abs/2312.00407)] [[Code](https://github.com/OpenLMLab/collie)]
- An Efficient 2D Method for Training Super-Large Deep Learning Models, <ins>IPDPS, 2023</ins> [[Paper](https://ieeexplore.ieee.org/document/10177476)] [[Code](https://github.com/xuqifan897/Optimus)]
- PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel, <ins>VLDB, 2023</ins> [[Paper](https://dl.acm.org/doi/10.14778/3611540.3611569)]
- Bamboo: Making Preemptible Instances Resilient for Affordable Training, <ins>NSDI, 2023</ins> [[Paper](https://www.usenix.org/system/files/nsdi23-thorpe.pdf)] [[Code](https://github.com/uclasystem/bamboo)]
- Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates, <ins>SOSP, 2023</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3600006.3613152)] [[Code](https://github.com/SymbioticLab/Oobleck)]
- Varuna: Scalable, Low-cost Training of Massive Deep Learning Models, <ins>EuroSys, 2022</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3492321.3519584)] [[Code](https://github.com/microsoft/varuna)]
- Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization, <ins>OSDI, 2022</ins> [[Paper](https://www.usenix.org/conference/osdi22/presentation/unger)] [[Code](https://github.com/flexflow/flexflow)]
- Tesseract: Parallelize the Tensor Parallelism Efficiently, <ins>ICPP, 2022</ins>, [[Paper](https://dl.acm.org/doi/10.1145/3545008.3545087)]
- Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning, <ins>OSDI, 2022</ins>, [[Paper](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)][[Code](https://github.com/alpa-projects/alpa)]
- Maximizing Parallelism in Distributed Training for Huge Neural Networks, <ins>arXiv, 2021</ins> [[Paper](https://arxiv.org/abs/2105.14450)]
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism, <ins>arXiv, 2020</ins> [[Paper](https://arxiv.org/pdf/1909.08053.pdf)]
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, <ins>SC, 2021</ins> [[Paper](https://dl.acm.org/doi/10.1145/3458817.3476209)] [[Code](https://github.com/nvidia/megatron-lm)]
- ZeRO-Infinity: breaking the GPU memory wall for extreme scale deep learning, <ins>SC, 2021</ins> [[Paper](https://dl.acm.org/doi/abs/10.1145/3458817.3476205)]
- ZeRO-Offload: Democratizing Billion-Scale Model Training, <ins>USENIX ATC, 2021</ins> [[Paper](https://www.usenix.org/system/files/atc21-ren-jie.pdf)] [[Code](https://www.deepspeed.ai/tutorials/zero-offload/)]
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, <ins>SC, 2020</ins> [[Paper](https://dl.acm.org/doi/10.5555/3433701.3433727)] [[Code](https://github.com/microsoft/DeepSpeed)]

#### System-Level Serving Efficiency Optimization
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
