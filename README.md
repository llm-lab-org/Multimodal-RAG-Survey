# Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.08826)

This repository is designed to collect and categorize papers related to Multimodal Retrieval-Augmented Generation (RAG) according to our survey paper: [Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2502.08826). Given the rapid growth in this field, we will continuously update both the paper and this repository to serve as a resource for researchers working on future projects.


## 📑 List of Contents


  - [🔎 General Pipeline](#-general-pipeline)
  - [🌿 Taxonomy of Recent Advances and Enhancements](#-Taxonomy-of-Recent-Advances-and-Enhancements)
  - [⚙ Taxonomy of Application Domains](#-Taxonomy-of-Application-Domains)
  - [📝 Abstract](#-Abstract)
  - [📊 Overview of Popular Datasets](#-Overview-of-Popular-Datasets)
    - [🖼 Image-Text](#-Image-Text)
    - [🎞 Video-Text](#-Video-Text)
    - [🔊 Audio-Text](#-Audio-Text)
    - [🩺 Medical](#-Medical)
    - [👗 Fashion](#-Fashion)
    - [🩺 Medical](#-Medical)
    - [💡 QA](#-QA)
    - [🌎 Other](#-Other)
  - [📄 Papers](#-Papers)
    - [📚 RAG-related Surveys](#-RAG-related-Surveys)
    - [👓 Retrieval Strategies Advances](#-retrieval-strategies-advances)
      - [🔍 Efficient-Search and Similarity Retrieval](#-Efficient-Search-and-Similarity-Retrieval)
        - [❓ Maximum Inner Product Search-MIPS](#-Maximum-Inner-Product-Search-MIPS)
        - [💫 Multimodal Encoders](#-Multimodal-Encoders)
      - [🎨 Modality-Centric Retrieval](#-Modality-Centric-Retrieval)
        - [📋 Text-Centric](#-Text-Centric)
        - [📸 Vision-Centric](#-Vision-Centric)
        - [🎥 Video-Centric](#-Video-Centric)
        - [📰 Document Retrieval and Layout Understanding](#-Document-Retrieval-and-Layout-Understanding)
      - [🥇🥈 Re-ranking Strategies](#-Re-ranking-Strategies)
        - [🎯 Optimized Example Selection](#-Optimized-Example-Selection)
        - [🧮 Relevance Score Evaluation](#-Relevance-Score-Evaluation)
        - [⏳ Filtering Mechanisms](#-Filtering-Mechanisms)
    - [🛠 Fusion Mechanisms](#-Fusion-Mechanisms)
      - [🎰 Score Fusion and Alignment](#-Score-Fusion-and-Alignment)
      - [⚔ Attention-Based Mechanisms](#-Attention-Based-Mechanisms)
      - [🧩 Unified Frameworkes](#-Unified-Frameworkes)
    - [🚀 Augmentation Techniques](#-Augmentation-Techniques)
      - [💰 Context-Enrichment](#-Context-Enrichment)
      - [🎡 Adaptive and Iterative Retrieval](#-Adaptive-and-Iterative-Retrieval)
      - [🧩 Unified Frameworkes](#-Unified-Frameworkes)
    - [🤖 Generation Techniques](#-Generation-Techniques)
      - [🧠 In-Context Learning](#-In-Context-Learning)
      - [👨‍⚖️ Reasoning](#-Reasoning)
      - [🤺 Instruction Tuning](#-Instruction-Tuning)
      - [📂 Source Attribution and Evidence Transparency](#-Source-Attribution-and-Evidence-Transparency)
    - [🔧 Training Strategies and Loss Function](#-Training-Strategies-and-Loss-Function)     
    - [🛡️ Robustness and Noise Management](#-Robustness-and-Noise-Management)            
  - [🔗 Citations](#-Citations)
  - [📧 Contact](#-Contact)

---
## 🔎 General Pipeline
![output-onlinepngtools (1)](https://github.com/user-attachments/assets/fabab7c0-9ca3-4d0b-b4d5-fb46defc8620)

## 🌿 Taxonomy of Recent Advances and Enhancements
![6634_Ask_in_Any_Modality_A_Com_organized-1-cropped](https://github.com/user-attachments/assets/0b5cd8e6-1aef-402b-a0a3-e3bf5cf555ae)

## ⚙ Taxonomy of Application Domains
![6634_Ask_in_Any_Modality_A_Com_organized-2-cropped](https://github.com/user-attachments/assets/f46ac78b-f51a-43c4-90bc-938d441093f2)


## 📝 Abstract
Large Language Models (LLMs) struggle with hallucinations and outdated knowledge due to their reliance on static training data. Retrieval-Augmented Generation (RAG) mitigates these issues by integrating external dynamic information enhancing factual and updated grounding. Recent advances in multimodal learning have led to the development of Multimodal RAG, incorporating multiple modalities such as text, images, audio, and video to enhance the generated outputs. However, cross-modal alignment and reasoning introduce unique challenges to Multimodal RAG, distinguishing it from traditional unimodal RAG.

This survey offers a structured and comprehensive analysis of Multimodal RAG systems, covering datasets, metrics, benchmarks, evaluation, methodologies, and innovations in retrieval, fusion, augmentation, and generation. We precisely review training strategies, robustness enhancements, and loss functions, while also exploring the diverse Multimodal RAG scenarios. 
Furthermore, we discuss open challenges and future research directions to support advancements in this evolving field. This survey lays the foundation for developing more capable and reliable AI systems that effectively leverage multimodal dynamic external knowledge bases. 

## 📊 Overview of Popular Datasets

### 🖼 Image-Text 

| **Name**         | **Statistics and Description**                                                                 | **Modalities** | **Link**                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------|----------------|-----------------------------------------------------------------------------------------------------|
| LAION-400M      | 200M image–text pairs; used for pre-training multimodal models.                                 | Image, Text    | [LAION-400M](https://laion.ai/projects/laion-400-mil-open-dataset/)                                 |
| Conceptual-Captions (CC) | 15M image–caption pairs; multilingual English–German image descriptions.                       | Image, Text    | [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions)             |
| CIRR            | 36,554 triplets from 21,552 images; focuses on natural image relationships.                    | Image, Text    | [CIRR](https://github.com/Cuberick-Orion/CIRR)                                                      |
| MS-COCO         | 330K images with captions; used for caption-to-image and image-to-caption generation.          | Image, Text    | [MS-COCO](https://cocodataset.org/)                                                                 |
| Flickr30K       | 31K images annotated with five English captions per image.                                     | Image, Text    | [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)                                      |
| Multi30K        | 30K German captions from native speakers and human-translated captions.                        | Image, Text    | [Multi30K](https://github.com/multi30k/dataset)                                                    |
| NoCaps          | For zero-shot image captioning evaluation; 15K images.                                         | Image, Text    | [NoCaps](https://nocaps.org/)                                                                       |
| Laion-5B        | 5B image–text pairs used as external memory for retrieval.                                     | Image, Text    | [LAION-5B](https://laion.ai/blog/laion-5b/)                                                        |
| COCO-CN         | 20,341 images for cross-lingual tagging and captioning with Chinese sentences.                 | Image, Text    | [COCO-CN](https://github.com/li-xirong/coco-cn)                                                    |
| CIRCO           | 1,020 queries with an average of 4.53 ground truths per query; for composed image retrieval.   | Image, Text    | [CIRCO](https://github.com/miccunifi/CIRCO)                                                        |

---

### 🎞 Video-Text

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| BDD-X           | 77 hours of driving videos with expert textual explanations; for explainable driving behavior.                  | Video, Text      | [BDD-X](https://github.com/JinkyuKimUCB/BDD-X-dataset)                                              |
| YouCook2        | 2,000 cooking videos with aligned descriptions; focused on video–text tasks.                                   | Video, Text      | [YouCook2](https://youcook2.eecs.umich.edu/)                                                        |
| ActivityNet     | 20,000 videos with multiple captions; used for video understanding and captioning.                              | Video, Text      | [ActivityNet](http://activity-net.org/)                                                             |
| SoccerNet       | Videos and metadata for 550 soccer games; includes transcribed commentary and key event annotations.            | Video, Text      | [SoccerNet](https://www.soccer-net.org/)                                                            |
| MSR-VTT         | 10,000 videos with 20 captions each; a large video description dataset.                                         | Video, Text      | [MSR-VTT](https://ms-multimedia-challenge.com/2016/dataset)                                         |
| MSVD            | 1,970 videos with approximately 40 captions per video.                                                         | Video, Text      | [MSVD](https://www.cs.utexas.edu/~ml/clamp/videoDescription/)                                       |
| LSMDC           | 118,081 video–text pairs from 202 movies; a movie description dataset.                                         | Video, Text      | [LSMDC](https://sites.google.com/site/describingmovies/)                                            |
| DiDemo          | 10,000 videos with four concatenated captions per video; with temporal localization of events.                  | Video, Text      | [DiDemo](https://github.com/LisaAnne/TemporalLanguageRelease)                                       |
| Breakfast            | 1,712 videos of breakfast preparation; one of the largest fully annotated video datasets.                       | Video, Text      | [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/)                   |
| COIN                 | 11,827 instructional YouTube videos across 180 tasks; for comprehensive instructional video analysis.            | Video, Text      | [COIN](https://coin-dataset.github.io/)                                                             |
| MSRVTT-QA            | Video question answering benchmark.                                                                             | Video, Text      | [MSRVTT-QA](https://github.com/xudejing/video-question-answering)                                   |
| MSVD-QA              | 1,970 video clips with approximately 50.5K QA pairs; video QA dataset.                                          | Video, Text      | [MSVD-QA](https://github.com/xudejing/video-question-answering)                                     |
| ActivityNet-QA       | 58,000 human–annotated QA pairs on 5,800 videos; benchmark for video QA models.                                 | Video, Text      | [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa)                                          |
| EpicKitchens-100     | 700 videos (100 hours of cooking activities) for online action prediction; egocentric vision dataset.           | Video, Text      | [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2021/)                                         |
| Ego4D                | 4.3M video–text pairs for egocentric videos; massive-scale egocentric video dataset.                            | Video, Text      | [Ego4D](https://ego4d-data.org/)                                                                    |
| HowTo100M            | 136M video clips with captions from 1.2M YouTube videos; for learning text–video embeddings.                    | Video, Text      | [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)                                       |
| CharadesEgo          | 68,536 activity instances from ego–exo videos; used for evaluation.                                             | Video, Text      | [Charades-Ego](https://prior.allenai.org/projects/charades-ego)                                     |
| ActivityNet Captions | 20K videos with 3.7 temporally localized sentences per video; dense-captioning events in videos.                 | Video, Text      | [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/)                      |
| VATEX                | 34,991 videos, each with multiple captions; a multilingual video-and-language dataset.                          | Video, Text      | [VATEX](https://eric-xw.github.io/vatex-website/)                                                   |
| Charades             | 9,848 video clips with textual descriptions; a multimodal research dataset.                                     | Video, Text      | [Charades](https://allenai.org/plato/charades/)                                                     |
| WebVid               | 10M video–text pairs (refined to WebVid-Refined-1M).                                                            | Video, Text      | [WebVid](https://github.com/m-bain/webvid)                                                          |
| Youku-mPLUG          | Chinese dataset with 10M video–text pairs (refined to Youku-Refined-1M).                                        | Video, Text      | [Youku-mPLUG](https://github.com/X-PLUG/Youku-mPLUG)   

---

### 🔊 Audio-Text

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| LibriSpeech     | 1,000 hours of read English speech with corresponding text; ASR corpus based on audiobooks.                     | Audio, Text      | [LibriSpeech](https://www.openslr.org/12)                                                           |
| SpeechBrown     | 55K paired speech-text samples; 15 categories covering diverse topics from religion to fiction.                 | Audio, Text      | [SpeechBrown](https://huggingface.co/datasets/llm-lab/SpeechBrown)                                   |
| AudioCap        | 46K audio clips paired with human-written text captions.                                                       | Audio, Text      | [AudioCaps](https://audiocaps.github.io/)                                                           |
| AudioSet        | 2M human-labeled sound clips from YouTube across diverse audio event classes (e.g., music or environmental).     | Audio            | [AudioSet](https://research.google.com/audioset/)                                                   |

---

### 🩺 Medical

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| MIMIC-CXR       | 125,417 labeled chest X-rays with reports; widely used for medical imaging research.                            | Image, Text      | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)                                         |
| CheXpert        | 224,316 chest radiographs of 65,240 patients; focused on medical analysis.                                      | Image, Text      | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)                               |
| MIMIC-III       | Health-related data from over 40K patients; includes clinical notes and structured data.                        | Text             | [MIMIC-III](https://mimic.physionet.org/)                                                           |
| IU-Xray         | 7,470 pairs of chest X-rays and corresponding diagnostic reports.                                               | Image, Text      | [IU-Xray](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)                   |
| PubLayNet       | 100,000 training samples and 2,160 test samples built from PubLayNet for document layout analysis.              | Image, Text      | [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)                                               |

---

### 👗 Fashion

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| Fashion-IQ       | 77,684 images across three categories; evaluated with Recall@10 and Recall@50 metrics.                         | Image, Text      | [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq)                                             |
| FashionGen       | 260.5K image–text pairs of fashion images and item descriptions.                                               | Image, Text      | [FashionGen](https://www.elementai.com/datasets/fashiongen)                                         |
| VITON-HD         | 83K images for virtual try-on; high-resolution clothing items dataset.                                         | Image, Text      | [VITON-HD](https://github.com/shadow2496/VITON-HD)                                                  |
| Fashionpedia     | 48,000 fashion images annotated with segmentation masks and fine-grained attributes.                           | Image, Text      | [Fashionpedia](https://fashionpedia.ai/)                                                            |
| DeepFashion      | Approximately 800K diverse fashion images for pseudo triplet generation.                                       | Image, Text      | [DeepFashion](https://github.com/zalandoresearch/fashion-mnist)                                     |

---

### 💡 QA

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| VQA              | 400K QA pairs with images for visual question-answering tasks.                                                 | Image, Text      | [VQA](https://visualqa.org/)                                                                        |
| PAQ              | 65M text-based QA pairs; a large-scale dataset for open-domain QA tasks.                                       | Text             | [PAQ](https://github.com/facebookresearch/PAQ)                                                      |
| ELI5             | 270K complex questions augmented with web pages and images; designed for long-form QA tasks.                   | Text             | [ELI5](https://facebookresearch.github.io/ELI5/)                                                    |
| OK-VQA           | 14K questions requiring external knowledge for visual question answering tasks.                                | Image, Text      | [OK-VQA](https://okvqa.allenai.org/)                                                                |
| WebQA            | 46K queries requiring reasoning across text and images; multimodal QA dataset.                                 | Text, Image      | [WebQA](https://webqna.github.io/)                                                                  |
| Infoseek         | Fine-grained visual knowledge retrieval using a Wikipedia-based knowledge base (~6M passages).                 | Image, Text      | [Infoseek](https://open-vision-language.github.io/infoseek/)                                        |
| ClueWeb22        | 10 billion web pages organized into subsets; a large-scale web corpus for retrieval tasks.                     | Text             | [ClueWeb22](https://lemurproject.org/clueweb22/)                                                    |
| MOCHEG           | 15,601 claims annotated with truthfulness labels and accompanied by textual and image evidence.                | Text, Image      | [MOCHEG](https://github.com/VT-NLP/Mocheg)                                                          |
| VQA v2           | 1.1M questions (augmented with VG-QA questions) for fine-tuning VQA models.                                    | Image, Text      | [VQA v2](https://visualqa.org/)                                                                     |    
| A-OKVQA          | Benchmark for visual question answering using world knowledge; around 25K questions.                          | Image, Text      | [A-OKVQA](https://github.com/allenai/aokvqa)                                                               |
| XL-HeadTags      | 415K news headline-article pairs spanning 20 languages across six diverse language families.                    | Text             | [XL-HeadTags](https://huggingface.co/datasets/faisaltareque/XL-HeadTags)                            |
| SEED-Bench       | 19K multiple-choice questions with accurate human annotations across 12 evaluation dimensions.                 | Text             | [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench)                                               |

---

### 🌎 Other

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| ImageNet         | 14M labeled images across thousands of categories; used as a benchmark in computer vision research.             | Image            | [ImageNet](http://www.image-net.org/)                                                               |
| Oxford Flowers102| Dataset of flowers with 102 categories for fine-grained image classification tasks.                            | Image            | [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)                            |
| Stanford Cars    | Images of different car models (five examples per model); used for fine-grained categorization tasks.           | Image            | [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)                |
| GeoDE            | 61,940 images from 40 classes across six world regions; emphasizes geographic diversity in object recognition.   | Image            | [GeoDE](https://github.com/AliRamazani/GeoDE)                                                       |

---

## 📄 Papers
### 📚 RAG-related Surveys
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) 
- [Old IR Methods Meet RAG](https://dl.acm.org/doi/pdf/10.1145/3626772.3657935)  
- [A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110)  
- [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921)  
- [A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2405.06211)
- [RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing](https://arxiv.org/abs/2404.19543)  
- [Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make Your LLMs Use External Data More Wisely](https://arxiv.org/abs/2409.14924)  
- [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)  
- [Retrieval-Augmented Generation for Natural Language Processing: A Survey](https://arxiv.org/abs/2407.13193)  
- [A Survey on Retrieval-Augmented Text Generation for Large Language Models](https://arxiv.org/abs/2404.10981)  
- [Graph Retrieval-Augmented Generation for Large Language Models: A Survey](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4895062) 
- [Trustworthiness in Retrieval-Augmented Generation Systems: A Survey](https://arxiv.org/abs/2409.10102)  
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136)


### 👓 Retrieval Strategies Advances
#### 🔍 Efficient-Search and Similarity Retrieval
##### ❓ Maximum Inner Product Search-MIPS
- [ADQ: Adaptive Dataset Quantization](https://arxiv.org/abs/2412.16895)
- [Query-Aware Quantization for Maximum Inner Product Search](https://ojs.aaai.org/index.php/AAAI/article/view/25613)
- [TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://papers.nips.cc/paper_files/paper/2022/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract-Conference.html)
- [ScaNN: Accelerating large-scale inference with anisotropic vector quantization](https://dl.acm.org/doi/abs/10.5555/3524938.3525302)
- [BanditMIPS: Faster Maximum Inner Product Search in High Dimensions](https://openreview.net/forum?id=FKkkdyRdsD)
- [MUST: An Effective and Scalable Framework for Multimodal Search of Target Modality](https://arxiv.org/abs/2312.06397)
- [FARGO: Fast Maximum Inner Product Search via Global Multi-Probing](https://dl.acm.org/doi/10.14778/3579075.3579084)
- [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928)
- [RA-CM3: Retrieval-Augmented Multimodal Language Modeling](https://proceedings.mlr.press/v202/yasunaga23a.html)
- [Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based Approximate Nearest Neighbor Search](https://arxiv.org/abs/2410.20381)
- [Revisiting Neural Retrieval on Accelerators](https://dl.acm.org/doi/10.1145/3580305.3599897)
- [DeeperImpact: Optimizing Sparse Learned Index Structures](https://arxiv.org/abs/2405.17093)
- [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](https://arxiv.org/abs/2409.10516)
- [Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation](https://arxiv.org/abs/2407.15268)

##### 💫 Multi-Modal Encoders
- [Mi-RAG: Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/)
- [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](https://arxiv.org/abs/2405.20797)
- [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855)
- [MARVEL: Unlocking the Multi-Modal Capability of Dense Retrieval via Visual Module Plugin](https://aclanthology.org/2024.acl-long.783/)
- [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://aclanthology.org/2024.acl-long.175/)
- [InternVideo: General Video Foundation Models via Generative and Discriminative Learning](https://arxiv.org/abs/2212.03191)
- [UniIR: Training and Benchmarking Universal Multimodal Information Retrievers](https://arxiv.org/abs/2311.17136)
- [UniVL-DR: Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval](https://arxiv.org/abs/2209.00179)
- [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482)
- [ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://proceedings.mlr.press/v139/jia21b.html)
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

#### 🎨 Modality-Centric Retrieval
##### 📋 Text-Centric
- [M<sup>2</sup>RAG: Multi-modal Retrieval Augmented Multi-modal Generation: Datasets, Evaluation Metrics and Strong Baselines](https://arxiv.org/abs/2411.16365)
- [CRAG: Corrective Retrieval Augmented Generation](https://openreview.net/forum?id=JnWJbrnaUE)
- [RAFT: Adapting Language Model to Domain Specific RAG](https://openreview.net/forum?id=rzQGHXNReU#discussion)
- [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://aclanthology.org/2024.acl-long.289/)
- [CRAG: Corrective Retrieval Augmented Generation](https://openreview.net/forum?id=JnWJbrnaUE)
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/)
- [XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags](https://aclanthology.org/2024.findings-acl.771/)
- [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)
- [Re-Imagen: Retrieval-Augmented Text-to-Image Generator](https://arxiv.org/abs/2209.14491)
- [GTE: Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)
- [Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118)

##### 📸 Vision-Centric
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/abs/2412.14457)
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927)
- [EchoSight: Advancing Visual-Language Models with Wiki Knowledge](https://aclanthology.org/2024.findings-emnlp.83/)
- [XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags](https://aclanthology.org/2024.findings-acl.771/)
- [RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training](https://dl.acm.org/doi/10.1145/3581783.3611830)
- [eCLIP: Improving Medical Multi-modal Contrastive Learning with Expert Annotations](https://link.springer.com/chapter/10.1007/978-3-031-72661-3_27)
- [Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval](https://arxiv.org/abs/2302.03084)
- [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://arxiv.org/abs/2404.15516)
- [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](https://aclanthology.org/2024.emnlp-main.89/)
- [VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering](https://arxiv.org/abs/2312.12273)

##### 🎥 Video-Centric
- [VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos](https://arxiv.org/abs/2502.01549)
- [VideoRAG: Retrieval-Augmented Generation over Video Corpus](https://arxiv.org/abs/2501.05874)
- [iRAG: Advancing RAG for Videos with an Incremental Approach](https://dl.acm.org/doi/10.1145/3627673.3680088)
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093)
- [Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval](https://arxiv.org/abs/2403.17998)
- [MV-Adapter: Multimodal Video Transfer Learning for Video Text Retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Jin_MV-Adapter_Multimodal_Video_Transfer_Learning_for_Video_Text_Retrieval_CVPR_2024_paper.pdf)
- [OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer](https://aclanthology.org/2024.emnlp-main.559/)
- [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610)
- [CTCH: Contrastive Transformer Cross-Modal Hashing for Video-Text Retrieval](https://www.ijcai.org/proceedings/2024/136)
- [Reversed in Time: A Novel Temporal-Emphasized Benchmark for Cross-Modal Video-Text Retrieval](https://dl.acm.org/doi/10.1145/3664647.3680731)
- [DrVideo: Document Retrieval Based Long Video Understanding](https://arxiv.org/abs/2406.12846)

##### 📰 Document Retrieval and Layout Understanding
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
- [ColQwen2: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://arxiv.org/abs/2411.04952)
- [ViTLP: Visually Guided Generative Text-Layout Pre-training for Document Intelligence](https://aclanthology.org/2024.naacl-long.264/)
- [DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding](https://aclanthology.org/2024.acl-long.463/)
- [CREAM: Coarse-to-Fine Retrieval and Multi-modal Efficient Tuning for Document VQA](https://dl.acm.org/doi/10.1145/3664647.3680750)
- [mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding](https://aclanthology.org/2024.findings-emnlp.175/)
- [mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](https://arxiv.org/abs/2409.03420)
- [VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2412.10704)
- [DSE: Unifying Multimodal Retrieval via Document Screenshot Embedding](https://arxiv.org/abs/2406.11251)
- [Robust Multi Model RAG Pipeline For Documents Containing Text, Table & Images](https://www.semanticscholar.org/paper/Robust-Multi-Model-RAG-Pipeline-For-Documents-Text%2C-Joshi-Gupta/282d9048c524eb3d87f73a3fe5ef49bc7297e8b4)
- [LoCAL: LoRA-Contextualizing Adaptation of Large Multimodal Models for Long Document Understanding](https://arxiv.org/abs/2411.01106)


#### 🥇🥈 Re-ranking Strategies
##### 🎯 Optimized Example Selection
- [MSIER: How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?](https://aclanthology.org/2024.emnlp-main.305/)
- [Hybrid RAG-empowered Multi-modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-based Contract Approach](https://arxiv.org/abs/2407.00978)
- [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://aclanthology.org/2024.emnlp-main.62/)
- [RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training](https://dl.acm.org/doi/10.1145/3581783.3611830)
- [M2-RAAP: A Multi-Modal Recipe for Advancing Adaptation-based Pre-training towards Effective and Efficient Zero-shot Video-text Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657833)
  
##### 🧮 Relevance Score Evaluation
- [RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performance](https://arxiv.org/abs/2501.03995)
- [Re-ranking the Context for Multimodal Retrieval Augmented Generation](https://arxiv.org/abs/2501.04695)
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103)
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://arxiv.org/abs/2411.15041)
- [LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657740)
- [RAGTrans: Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction](https://dl.acm.org/doi/10.1145/3637528.3672041)
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/)
- [EgoInstructor: Retrieval-Augmented Egocentric Video Captioning](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Retrieval-Augmented_Egocentric_Video_Captioning_CVPR_2024_paper.pdf)



##### ⏳ Filtering Mechanisms



### 🛠 Fusion Mechanisms
#### 🎰 Score Fusion and Alignment
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103)
- [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311)
- [Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications](https://arxiv.org/abs/2410.21943)
- [Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs](https://arxiv.org/abs/2404.15406)
- [C3Net: Compound Conditioned ControlNet for Multimodal Content Generation](https://arxiv.org/abs/2311.17951)
- [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](https://arxiv.org/abs/2412.14475)
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://aclanthology.org/2024.emnlp-demo.33/)
- [REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory](https://arxiv.org/abs/2212.05221)
- [MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](https://arxiv.org/abs/2404.05726)
- [Large Language Models Know What is Key Visual Entity: An LLM-assisted Multimodal Retrieval for VQA](https://aclanthology.org/2024.emnlp-main.613/)
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154)
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927)
- [Re-Imagen: Retrieval-Augmented Text-to-Image Generator](https://arxiv.org/abs/2209.14491)
- [VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents](https://arxiv.org/abs/2410.10594)
- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828)
#### ⚔ Attention-Based Mechanisms
- [EMERGE: Integrating RAG for Improved Multimodal EHR Predictive Modeling](https://arxiv.org/abs/2406.00036)
- [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://arxiv.org/abs/2402.13625)
- [AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles](https://arxiv.org/pdf/2412.16701)
- [RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training](https://arxiv.org/abs/2303.00534)
- [Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction](https://dl.acm.org/doi/abs/10.1145/3637528.3672041)
- [MV-Adapter: Multimodal Video Transfer Learning for Video Text Retrieval](https://arxiv.org/abs/2301.07868)
- [M<sup>2-RAAP: A Multi-Modal Recipe for Advancing Adaptation-based Pre-training towards Effective and Efficient Zero-shot Video-text Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657833)
- [Retrieval-augmented egocentric video captioning](https://arxiv.org/abs/2401.00789)
- [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928)
- [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610)
#### 🧩 Unified Frameworkes
- [Hybrid RAG-Empowered Multi-Modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-Based Contract Approach](https://arxiv.org/abs/2407.00978)
- [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](https://www.techrxiv.org/doi/full/10.36227/techrxiv.172840252.24352951)
- [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://arxiv.org/pdf/2411.04952)
- [PDF-MVQA: A Dataset for Multimodal Information Retrieval in PDF-based Visual Question Answering](https://arxiv.org/abs/2404.12720)
- [Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval](https://arxiv.org/abs/2404.15875)
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321v1)
- [UFineBench: Towards Text-based Person Retrieval with Ultra-fine Granularity - CVPR 2024](https://arxiv.org/abs/2312.03441)
- [Multimodal Learned Sparse Retrieval with Probabilistic Expansion Control](https://arxiv.org/abs/2402.17535)

### 🚀 Augmentation Techniques
#### 💰 Context-Enrichment 
- [EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation](https://doi.org/10.1145/3627673.3679582)  
- [Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/)  
- [Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs](https://openaccess.thecvf.com/content/CVPR2024/html/Caffagni_Wiki-LLaVA_Hierarchical_Retrieval-Augmented_Generation_for_Multimodal_LLMs_CVPR_2024_paper.html)  
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093)  
- [Img2Loc: Revisiting Image Geolocalization Using Multi-Modality Foundation Models and Image-Based Retrieval-Augmented Generation](https://doi.org/10.1145/3627673.3679582)  
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927) 

#### 🎡 Adaptive and Iterative Retrieval
- [Enhancing Multi-modal Multi-hop Question Answering via Structured Knowledge and Unified Retrieval-Generation](https://doi.org/10.1145/3581783.3611964)  
- [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1)  
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/)  
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321)  
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085)  
- [Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent](https://arxiv.org/abs/2411.02937)  
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://api.semanticscholar.org/CorpusID:274192536)  
- [RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models](https://aclanthology.org/2024.fever-1.29/)


### 🤖 Generation Techniques
#### 🧠 In-Context Learning 
- [Retrieval Meets Reasoning: Even High-school Textbook Knowledge Benefits Multimodal Reasoning](https://arxiv.org/abs/2405.20834)  
- [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311)  
- [Retrieval-Augmented Multimodal Language Modeling (RA-CM3)](https://proceedings.mlr.press/v202/yasunaga23a.html)  
- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828)  
- [How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning? (MSIER)](https://aclanthology.org/2024.emnlp-main.305/)  
- [RAVEN: Multitask Retrieval Augmented Vision-Language Learning](https://arxiv.org/abs/2406.19150)

#### 👨‍⚖️ Reasoning 
- [RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models](https://aclanthology.org/2024.fever-1.29/)  
- [VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2412.10704)  
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://paperswithcode.com/paper/self-adaptive-multimodal-retrieval-augmented)  
- [LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657740)

#### 🤺 Instruction Tuning 
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154)  
- [InstructBLIP: towards general-purpose vision-language models with instruction tuning](https://dl.acm.org/doi/10.5555/3666122.3668264)  
- [Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning](https://arxiv.org/abs/2501.01120v1)  
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://arxiv.org/html/2411.15041)  
- [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training (RagVL)](https://arxiv.org/abs/2407.21439)  
- [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://arxiv.org/abs/2404.15516)
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085)
- [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](https://arxiv.org/abs/2412.14475)  
- [SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information](https://arxiv.org/abs/2409.14083)
- [Rule: Reliable multimodal rag for factuality in medical vision language models](https://arxiv.org/abs/2407.05131)

#### 📂 Source Attribution and Evidence Transparency 
- [MuRAR: A Simple and Effective Multimodal Retrieval and Answer Refinement Framework for Multimodal Question Answering](https://arxiv.org/abs/2408.08521)  
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/abs/2412.14457)  
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75.pdf)

### 🔧 Training Strategies and Loss Function
- [REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.html) (April 2023)
- [Multimodal Learned Sparse Retrieval with Probabilistic Expansion Control](https://arxiv.org/abs/2402.17535) (February 2024)
- [HACL: Hallucination Augmented Contrastive Learning for Multimodal Large Language Model](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Hallucination_Augmented_Contrastive_Learning_for_Multimodal_Large_Language_Model_CVPR_2024_paper.html) (February 2024)  
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103) (May 2024)  
- [Improving Medical Multi-modal Contrastive Learning with Expert Annotations](https://doi.org/10.1007/978-3-031-72661-3_27) (November 2024)
- [EchoSight: Advancing Visual-Language Models with Wiki Knowledge](https://aclanthology.org/2024.findings-emnlp.83/) (November 2024)

### 🛡️ Robustness and Noise Management
- [RA-CM3: Retrieval-Augmented Multimodal Language Modeling](https://arxiv.org/abs/2211.12561) (January 2023)
- [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training](https://arxiv.org/abs/2407.21439) (July 2024)  
- [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://aclanthology.org/2024.findings-acl.69/) (August 2024)
- [RAGTrans: Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction](https://doi.org/10.1145/3637528.3672041) (August 2024)
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154) (October 2024)  
- [Quantifying the Gaps Between Translation and Native Perception in Training for Multimodal, Multilingual Retrieval](https://aclanthology.org/2024.emnlp-main.335/) (November 2024)  
- [AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles](https://arxiv.org/abs/2412.16701) (December 2024)  

**This README is a work in progress and will be completed soon. Stay tuned for more updates!**

---
## 🔗 Citations
If you find our paper, code, data, or models useful, please cite the paper:
```
@misc{abootorabi2025askmodalitycomprehensivesurvey,
      title={Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation}, 
      author={Mohammad Mahdi Abootorabi and Amirhosein Zobeiri and Mahdi Dehghani and Mohammadali Mohammadkhani and Bardia Mohammadi and Omid Ghahroodi and Mahdieh Soleymani Baghshah and Ehsaneddin Asgari},
      year={2025},
      eprint={2502.08826},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08826}, 
}
```
## 📧 Contact
If you have questions, please send an email to mahdi.abootorabi2@gmail.com.
