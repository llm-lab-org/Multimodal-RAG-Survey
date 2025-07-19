# Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.08826) [![Website](https://img.shields.io/website?url=https%3A%2F%2Fmultimodalrag.github.io%2F)](https://multimodalrag.github.io/)

This repository is designed to collect and categorize papers related to Multimodal Retrieval-Augmented Generation (RAG) according to our survey paper: [Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2502.08826). Given the rapid growth in this field, we will continuously update both the paper and this repository to serve as a resource for researchers working on future projects.

## 📢 News
- **June 2, 2025**: A new enhanced version of our paper is out now on arXiv! This update also includes new related papers and covers new topics such as agentic interaction and audio-centric retrieval.
- **May 15, 2025**: This paper has been accepted for publication in the **[ACL 2025](https://2025.aclweb.org/) Findings**.
- **April 18, 2025**: Our [website](https://multimodalrag.github.io/) for this topic is up now.  
- **February 17, 2025**: We release the first survey for Multimodal Retrieval-Augmented Generation.
*Feel free to cite, contribute, or open a pull request to add recent related papers!*

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
        - [🎶 Audio-Centric](#-Audio-Centric)
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
    - [🛠 Taks Addressed by Multimodal RAGs](#-Taks-Addressed-by-Multimodal-RAGs)
    - [📏 Evaluation Metrics](#-Evaluation-Metrics)           
  - [🔗 Citations](#-Citations)
  - [📧 Contact](#-Contact)

---
## 🔎 General Pipeline
![MM-RAG (1)](https://github.com/user-attachments/assets/a86b449e-cb84-4dd8-99c1-7bb8b4cb91d8)

## 🌿 Taxonomy of Recent Advances and Enhancements
![Multimodal_Retrieval_Augmented_Generation__A_Survey___acl_final_organized](https://github.com/user-attachments/assets/bf984875-d008-4b42-abcb-35781fe27278)

## ⚙ Taxonomy of Application Domains
![applications](https://github.com/user-attachments/assets/da3dbd6e-b647-487a-ab00-d01f8ac6cc91)



## 📝 Abstract
Large Language Models (LLMs) suffer from hallucinations and outdated knowledge due to their reliance on static training data. Retrieval-Augmented Generation (RAG) mitigates these issues by integrating external dynamic information for improved factual grounding. With advances in multimodal learning, Multimodal RAG extends this approach by incorporating multiple modalities such as text, images, audio, and video to enhance the generated outputs. However, cross-modal alignment and reasoning introduce unique challenges beyond those in unimodal RAG. 

This survey offers a structured and comprehensive analysis of Multimodal RAG systems, covering datasets, benchmarks, metrics, evaluation, methodologies, and innovations in retrieval, fusion, augmentation, and generation. We precisely review training strategies, robustness enhancements, loss functions, and agent-based approaches, while also exploring the diverse Multimodal RAG scenarios. In addition, we outline open challenges and future directions to guide research in this evolving field. This survey lays the foundation for developing more capable and reliable AI systems that effectively leverage multimodal dynamic external knowledge bases.

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
- [RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing](https://arxiv.org/abs/2404.19543) ![](https://img.shields.io/badge/date-2025.06-red)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) ![](https://img.shields.io/badge/date-2025.03-red)
- [Retrieval-Augmented Generation for Natural Language Processing: A Survey](https://arxiv.org/abs/2407.13193) ![](https://img.shields.io/badge/date-2025.03-red)
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) ![](https://img.shields.io/badge/date-2025.02-red)
- [Graph Retrieval-Augmented Generation for Large Language Models: A Survey](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4895062) ![](https://img.shields.io/badge/date-2024.12-red)
- [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921) ![](https://img.shields.io/badge/date-2024.09-red)
- [Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make Your LLMs Use External Data More Wisely](https://arxiv.org/abs/2409.14924) ![](https://img.shields.io/badge/date-2024.09-red)
- [Trustworthiness in Retrieval-Augmented Generation Systems: A Survey](https://arxiv.org/abs/2409.10102) ![](https://img.shields.io/badge/date-2024.09-red)
- [A Survey on Retrieval-Augmented Text Generation for Large Language Models](https://arxiv.org/abs/2404.10981) ![](https://img.shields.io/badge/date-2024.08-red) 
- [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219) ![](https://img.shields.io/badge/date-2024.07-red)
- [Old IR Methods Meet RAG](https://dl.acm.org/doi/pdf/10.1145/3626772.3657935) ![](https://img.shields.io/badge/date-2024.07-red)
- [A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2405.06211) ![](https://img.shields.io/badge/date-2024.06-red)
- [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) ![](https://img.shields.io/badge/date-2023.12-red)
- [A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110) ![](https://img.shields.io/badge/date-2022.02-red)


### 👓 Retrieval Strategies Advances
#### 🔍 Efficient-Search and Similarity Retrieval
##### ❓ Maximum Inner Product Search-MIPS
- [Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation](https://arxiv.org/abs/2407.15268) ![](https://img.shields.io/badge/date-2025.02-red)
- [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](https://arxiv.org/abs/2409.10516) ![](https://img.shields.io/badge/date-2024.12-red)
- [ADQ: Adaptive Dataset Quantization](https://arxiv.org/abs/2412.16895) ![](https://img.shields.io/badge/date-2024.12-red)
- [Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based Approximate Nearest Neighbor Search](https://arxiv.org/abs/2410.20381) ![](https://img.shields.io/badge/date-2024.10-red)
- [DeeperImpact: Optimizing Sparse Learned Index Structures](https://arxiv.org/abs/2405.17093) ![](https://img.shields.io/badge/date-2024.07-red)
- [MUST: An Effective and Scalable Framework for Multimodal Search of Target Modality](https://arxiv.org/abs/2312.06397) ![](https://img.shields.io/badge/date-2023.12-red)
- [BanditMIPS: Faster Maximum Inner Product Search in High Dimensions](https://openreview.net/forum?id=FKkkdyRdsD) ![](https://img.shields.io/badge/date-2023.09-red)
- [Revisiting Neural Retrieval on Accelerators](https://dl.acm.org/doi/10.1145/3580305.3599897) ![](https://img.shields.io/badge/date-2023.08-red)
- [Query-Aware Quantization for Maximum Inner Product Search](https://ojs.aaai.org/index.php/AAAI/article/view/25613) ![](https://img.shields.io/badge/date-2023.06-red)
- [RA-CM3: Retrieval-Augmented Multimodal Language Modeling](https://proceedings.mlr.press/v202/yasunaga23a.html) ![](https://img.shields.io/badge/date-2023.04-red)
- [FARGO: Fast Maximum Inner Product Search via Global Multi-Probing](https://dl.acm.org/doi/10.14778/3579075.3579084) ![](https://img.shields.io/badge/date-2023.01-red)
- [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928) ![](https://img.shields.io/badge/date-2022.10-red)
- [TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://papers.nips.cc/paper_files/paper/2022/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract-Conference.html) ![](https://img.shields.io/badge/date-2022.06-red)
- [ScaNN: Accelerating large-scale inference with anisotropic vector quantization](https://dl.acm.org/doi/abs/10.5555/3524938.3525302) ![](https://img.shields.io/badge/date-2020.07-red)


##### 💫 Multi-Modal Encoders
- [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855) ![](https://img.shields.io/badge/date-2025.04-red)
- [Mi-RAG: Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/) ![](https://img.shields.io/badge/date-2024.11-red)
- [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://aclanthology.org/2024.acl-long.175/) ![](https://img.shields.io/badge/date-2024.08-red)
- [MARVEL: Unlocking the Multi-Modal Capability of Dense Retrieval via Visual Module Plugin](https://aclanthology.org/2024.acl-long.783/) ![](https://img.shields.io/badge/date-2024.08-red)
- [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](https://arxiv.org/abs/2405.20797) ![](https://img.shields.io/badge/date-2024.06-red)
- [UniIR: Training and Benchmarking Universal Multimodal Information Retrievers](https://arxiv.org/abs/2311.17136) ![](https://img.shields.io/badge/date-2023.11-red)
- [UniVL-DR: Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval](https://arxiv.org/abs/2209.00179) ![](https://img.shields.io/badge/date-2023.02-red)
- [InternVideo: General Video Foundation Models via Generative and Discriminative Learning](https://arxiv.org/abs/2212.03191) ![](https://img.shields.io/badge/date-2022.12-red)
- [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482) ![](https://img.shields.io/badge/date-2022.03-red)
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) ![](https://img.shields.io/badge/date-2022.02-red)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) ![](https://img.shields.io/badge/date-2021.03-red)
- [ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://proceedings.mlr.press/v139/jia21b.html) ![](https://img.shields.io/badge/date-2021.02-red)

#### 🎨 Modality-Centric Retrieval
##### 📋 Text-Centric
- [M<sup>2</sup>RAG: Multi-modal Retrieval Augmented Multi-modal Generation: Datasets, Evaluation Metrics and Strong Baselines](https://arxiv.org/abs/2411.16365) ![](https://img.shields.io/badge/date-2025.05-red)
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/) ![](https://img.shields.io/badge/date-2024.11-red)
- [CRAG: Corrective Retrieval Augmented Generation](https://openreview.net/forum?id=JnWJbrnaUE) ![](https://img.shields.io/badge/date-2024.09-red)
- [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://aclanthology.org/2024.acl-long.289/) ![](https://img.shields.io/badge/date-2024.08-red)
- [XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags](https://aclanthology.org/2024.findings-acl.771/) ![](https://img.shields.io/badge/date-2024.08-red)
- [RAFT: Adapting Language Model to Domain Specific RAG](https://openreview.net/forum?id=rzQGHXNReU#discussion) ![](https://img.shields.io/badge/date-2024.07-red)
- [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216) ![](https://img.shields.io/badge/date-2024.06-red)
- [GTE: Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281) ![](https://img.shields.io/badge/date-2023.08-red)
- [Re-Imagen: Retrieval-Augmented Text-to-Image Generator](https://arxiv.org/abs/2209.14491) ![](https://img.shields.io/badge/date-2022.11-red)
- [Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118) ![](https://img.shields.io/badge/date-2022.08-red)


##### 📸 Vision-Centric
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927) ![](https://img.shields.io/badge/date-2024.12-red)
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/abs/2412.14457) ![](https://img.shields.io/badge/date-2024.12-red)
- [eCLIP: Improving Medical Multi-modal Contrastive Learning with Expert Annotations](https://link.springer.com/chapter/10.1007/978-3-031-72661-3_27) ![](https://img.shields.io/badge/date-2024.11-red)
- [EchoSight: Advancing Visual-Language Models with Wiki Knowledge](https://aclanthology.org/2024.findings-emnlp.83/) ![](https://img.shields.io/badge/date-2024.11-red)
- [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](https://aclanthology.org/2024.emnlp-main.89/) ![](https://img.shields.io/badge/date-2024.11-red)
- [XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags](https://aclanthology.org/2024.findings-acl.771/) ![](https://img.shields.io/badge/date-2024.08-red)
- [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://arxiv.org/abs/2404.15516) ![](https://img.shields.io/badge/date-2024.04-red)
- [VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering](https://arxiv.org/abs/2312.12273) ![](https://img.shields.io/badge/date-2023.12-red)
- [RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training](https://dl.acm.org/doi/10.1145/3581783.3611830) ![](https://img.shields.io/badge/date-2023.10-red)
- [Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval](https://arxiv.org/abs/2302.03084) ![](https://img.shields.io/badge/date-2023.05-red)

##### 🎥 Video-Centric
- [VideoRAG: Retrieval-Augmented Generation over Video Corpus](https://arxiv.org/abs/2501.05874) ![](https://img.shields.io/badge/date-2025.05-red)
- [VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos](https://arxiv.org/abs/2502.01549) ![](https://img.shields.io/badge/date-2025.02-red)
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093) ![](https://img.shields.io/badge/date-2024.12-red)
- [DrVideo: Document Retrieval Based Long Video Understanding](https://arxiv.org/abs/2406.12846) ![](https://img.shields.io/badge/date-2024.11-red)
- [OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer](https://aclanthology.org/2024.emnlp-main.559/) ![](https://img.shields.io/badge/date-2024.11-red)
- [Reversed in Time: A Novel Temporal-Emphasized Benchmark for Cross-Modal Video-Text Retrieval](https://dl.acm.org/doi/10.1145/3664647.3680731) ![](https://img.shields.io/badge/date-2024.10-red)
- [iRAG: Advancing RAG for Videos with an Incremental Approach](https://dl.acm.org/doi/10.1145/3627673.3680088) ![](https://img.shields.io/badge/date-2024.10-red)
- [CTCH: Contrastive Transformer Cross-Modal Hashing for Video-Text Retrieval](https://www.ijcai.org/proceedings/2024/136) ![](https://img.shields.io/badge/date-2024.08-red)
- [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610) ![](https://img.shields.io/badge/date-2024.04-red)
- [Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval](https://arxiv.org/abs/2403.17998) ![](https://img.shields.io/badge/date-2024.03-red)
- [MV-Adapter: Multimodal Video Transfer Learning for Video Text Retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Jin_MV-Adapter_Multimodal_Video_Transfer_Learning_for_Video_Text_Retrieval_CVPR_2024_paper.pdf) ![](https://img.shields.io/badge/date-2023.01-red)


##### 🎶 Audio-Centric
- [CA-CLAP: Retrieval Augmented Generation in Prompt-based Text-to-Speech Synthesis with Context-Aware Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2406.03714) ![](https://img.shields.io/badge/date-2024.06-red)
- [RECAP: Retrieval-Augmented Audio Captioning](https://arxiv.org/abs/2309.09836) ![](https://img.shields.io/badge/date-2024.06-red)
- [SpeechRAG: Speech Retrieval-Augmented Generation without Automatic Speech Recognition](https://arxiv.org/abs/2411.13093) ![](https://img.shields.io/badge/date-2025.01-red)
- [WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](https://arxiv.org/abs/2502.14727) ![](https://img.shields.io/badge/date-2025.02-red)
- [SEAL: Speech Embedding Alignment Learning for Speech Large Language Model with Retrieval-Augmented Generation](https://arxiv.org/abs/2502.02603) ![](https://img.shields.io/badge/date-2025.01-red)
- [Audiobox TTA-RAG: Improving Zero-Shot and Few-Shot Text-To-Audio with Retrieval-Augmented Generation](https://arxiv.org/abs/2411.05141) ![](https://img.shields.io/badge/date-2025.01-red)
- [DRCap: Decoding CLAP Latents with Retrieval-Augmented Generation for Zero-shot Audio Captioning](https://arxiv.org/abs/2410.09472) ![](https://img.shields.io/badge/date-2025.01-red)
- [P2PCAP: Enhancing Retrieval-Augmented Audio Captioning with Generation-Assisted Multimodal Querying and Progressive Learning](https://arxiv.org/abs/2410.10913) ![](https://img.shields.io/badge/date-2025.01-red)
- [LA-RAG:Enhancing LLM-based ASR Accuracy with Retrieval-Augmented Generation](https://arxiv.org/abs/2409.08597) ![](https://img.shields.io/badge/date-2024.09-red)
- [Contextual asr with retrieval augmented large language model.](https://doi.org/10.1109/ICASSP49660.2025.10890057) ![](https://img.shields.io/badge/date-2025.01-red)


##### 📰 Document Retrieval and Layout Understanding
- [SV-RAG: LoRA-Contextualizing Adaptation of MLLMs for Long Document Understanding](https://arxiv.org/abs/2411.01106) ![](https://img.shields.io/badge/date-2025.03-red)
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) ![](https://img.shields.io/badge/date-2025.02-red)
- [VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2412.10704) ![](https://img.shields.io/badge/date-2025.02-red)
- [DSE: Unifying Multimodal Retrieval via Document Screenshot Embedding](https://arxiv.org/abs/2406.11251) ![](https://img.shields.io/badge/date-2024.12-red)
- [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://arxiv.org/abs/2411.04952) ![](https://img.shields.io/badge/date-2024.11-red)
- [mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding](https://aclanthology.org/2024.findings-emnlp.175/) ![](https://img.shields.io/badge/date-2024.11-red)
- [CREAM: Coarse-to-Fine Retrieval and Multi-modal Efficient Tuning for Document VQA](https://dl.acm.org/doi/10.1145/3664647.3680750) ![](https://img.shields.io/badge/date-2024.10-red)
- [ColQwen2: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) ![](https://img.shields.io/badge/date-2024.10-red)
- [mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](https://arxiv.org/abs/2409.03420) ![](https://img.shields.io/badge/date-2024.09-red)
- [DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding](https://aclanthology.org/2024.acl-long.463/) ![](https://img.shields.io/badge/date-2024.08-red)
- [Robust Multi Model RAG Pipeline For Documents Containing Text, Table & Images](https://www.semanticscholar.org/paper/Robust-Multi-Model-RAG-Pipeline-For-Documents-Text%2C-Joshi-Gupta/282d9048c524eb3d87f73a3fe5ef49bc7297e8b4) ![](https://img.shields.io/badge/date-2024.06-red)
- [ViTLP: Visually Guided Generative Text-Layout Pre-training for Document Intelligence](https://aclanthology.org/2024.naacl-long.264/) ![](https://img.shields.io/badge/date-2024.06-red)


#### 🥇🥈 Re-ranking Strategies
##### 🎯 Optimized Example Selection
- [Hybrid RAG-empowered Multi-modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-based Contract Approach](https://arxiv.org/abs/2407.00978) ![](https://img.shields.io/badge/date-2024.12-red)
- [MSIER: How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?](https://aclanthology.org/2024.emnlp-main.305/) ![](https://img.shields.io/badge/date-2024.11-red)
- [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://aclanthology.org/2024.emnlp-main.62/) ![](https://img.shields.io/badge/date-2024.11-red)
- [M2-RAAP: A Multi-Modal Recipe for Advancing Adaptation-based Pre-training towards Effective and Efficient Zero-shot Video-text Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657833) ![](https://img.shields.io/badge/date-2024.07-red)
- [RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training](https://dl.acm.org/doi/10.1145/3581783.3611830) ![](https://img.shields.io/badge/date-2023.10-red)
  
##### 🧮 Relevance Score Evaluation
- [Re-ranking the Context for Multimodal Retrieval Augmented Generation](https://arxiv.org/abs/2501.04695) ![](https://img.shields.io/badge/date-2025.01-red)
- [RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performance](https://arxiv.org/abs/2501.03995) ![](https://img.shields.io/badge/date-2025.01-red)
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://arxiv.org/abs/2411.15041) ![](https://img.shields.io/badge/date-2024.11-red)
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/) ![](https://img.shields.io/badge/date-2024.11-red)
- [RAGTrans: Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction](https://dl.acm.org/doi/10.1145/3637528.3672041) ![](https://img.shields.io/badge/date-2024.08-red)
- [LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657740) ![](https://img.shields.io/badge/date-2024.07-red)
- [EgoInstructor: Retrieval-Augmented Egocentric Video Captioning](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Retrieval-Augmented_Egocentric_Video_Captioning_CVPR_2024_paper.pdf) ![](https://img.shields.io/badge/date-2024.06-red)
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103) ![](https://img.shields.io/badge/date-2024.05-red)


##### ⏳ Filtering Mechanisms
- [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855) ![](https://img.shields.io/badge/date-2025.04-red)
- [MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs](https://arxiv.org/abs/2411.02571) ![](https://img.shields.io/badge/date-2025.02-red)
- [MuRAR: A Simple and Effective Multimodal Retrieval and Answer Refinement Framework for Multimodal Question Answering](https://aclanthology.org/2025.coling-demos.13/) ![](https://img.shields.io/badge/date-2025.01-red)
- [MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](https://arxiv.org/abs/2501.00332) ![](https://img.shields.io/badge/date-2024.12-red)
- [RAFT: Adapting Language Model to Domain Specific RAG](https://openreview.net/forum?id=rzQGHXNReU#discussion) ![](https://img.shields.io/badge/date-2024.08-red)


### 🛠 Fusion Mechanisms
#### 🎰 Score Fusion and Alignment

- [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red)
- [VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents](https://arxiv.org/abs/2410.10594) ![](https://img.shields.io/badge/date-2025.03-red)
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927) ![](https://img.shields.io/badge/date-2024.12-red)
- [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](https://arxiv.org/abs/2412.14475) ![](https://img.shields.io/badge/date-2024.12-red)
- [Large Language Models Know What is Key Visual Entity: An LLM-assisted Multimodal Retrieval for VQA](https://aclanthology.org/2024.emnlp-main.613/) ![](https://img.shields.io/badge/date-2024.11-red)
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://aclanthology.org/2024.emnlp-demo.33/) ![](https://img.shields.io/badge/date-2024.11-red)
- [Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications](https://arxiv.org/abs/2410.21943) ![](https://img.shields.io/badge/date-2024.10-red)
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154) ![](https://img.shields.io/badge/date-2024.10-red)
- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828) ![](https://img.shields.io/badge/date-2024.05-red)
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103) ![](https://img.shields.io/badge/date-2024.05-red)
- [Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs](https://arxiv.org/abs/2404.15406) ![](https://img.shields.io/badge/date-2024.05-red)
- [MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](https://arxiv.org/abs/2404.05726) ![](https://img.shields.io/badge/date-2024.04-red)
- [C3Net: Compound Conditioned ControlNet for Multimodal Content Generation](https://arxiv.org/abs/2311.17951) ![](https://img.shields.io/badge/date-2023.11-red)
- [REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory](https://arxiv.org/abs/2212.05221) ![](https://img.shields.io/badge/date-2023.04-red)
- [Re-Imagen: Retrieval-Augmented Text-to-Image Generator](https://arxiv.org/abs/2209.14491) ![](https://img.shields.io/badge/date-2022.11-red)

#### ⚔ Attention-Based Mechanisms
- [AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles](https://arxiv.org/pdf/2412.16701) ![](https://img.shields.io/badge/date-2025.06-red)
- [EMERGE: Integrating RAG for Improved Multimodal EHR Predictive Modeling](https://arxiv.org/abs/2406.00036) ![](https://img.shields.io/badge/date-2025.02-red)
- [Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction](https://dl.acm.org/doi/abs/10.1145/3637528.3672041) ![](https://img.shields.io/badge/date-2024.08-red)
- [M<sup>2</sup>-RAAP: A Multi-Modal Recipe for Advancing Adaptation-based Pre-training towards Effective and Efficient Zero-shot Video-text Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657833) ![](https://img.shields.io/badge/date-2024.07-red)
- [Retrieval-augmented egocentric video captioning](https://arxiv.org/abs/2401.00789) ![](https://img.shields.io/badge/date-2024.06-red)
- [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://arxiv.org/abs/2402.13625) ![](https://img.shields.io/badge/date-2024.06-red)
- [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610) ![](https://img.shields.io/badge/date-2024.04-red)
- [MV-Adapter: Multimodal Video Transfer Learning for Video Text Retrieval](https://arxiv.org/abs/2301.07868) ![](https://img.shields.io/badge/date-2024.04-red)
- [RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training](https://arxiv.org/abs/2303.00534) ![](https://img.shields.io/badge/date-2023.03-red)
- [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928) ![](https://img.shields.io/badge/date-2022.10-red)

#### 🧩 Unified Frameworks
- [Hybrid RAG-Empowered Multi-Modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-Based Contract Approach](https://arxiv.org/abs/2407.00978) ![](https://img.shields.io/badge/date-2024.12-red)
- [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://arxiv.org/pdf/2411.04952) ![](https://img.shields.io/badge/date-2024.11-red)
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321v1) ![](https://img.shields.io/badge/date-2024.10-red)
- [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](https://www.techrxiv.org/doi/full/10.36227/techrxiv.172840252.24352951) ![](https://img.shields.io/badge/date-2024.10-red)
- [UFineBench: Towards Text-based Person Retrieval with Ultra-fine Granularity - CVPR 2024](https://arxiv.org/abs/2312.03441) ![](https://img.shields.io/badge/date-2024.06-red)
- [Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval](https://arxiv.org/abs/2404.15875) ![](https://img.shields.io/badge/date-2024.04-red)
- [PDF-MVQA: A Dataset for Multimodal Information Retrieval in PDF-based Visual Question Answering](https://arxiv.org/abs/2404.12720) ![](https://img.shields.io/badge/date-2024.04-red)
- [Multimodal Learned Sparse Retrieval with Probabilistic Expansion Control](https://arxiv.org/abs/2402.17535) ![](https://img.shields.io/badge/date-2024.02-red)

### 🚀 Augmentation Techniques
#### 💰 Context-Enrichment 
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927) ![](https://img.shields.io/badge/date-2024.12-red)
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093) ![](https://img.shields.io/badge/date-2024.12-red)
- [Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/) ![](https://img.shields.io/badge/date-2024.11-red) 
- [EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation](https://doi.org/10.1145/3627673.3679582) ![](https://img.shields.io/badge/date-2024.10-red)
- [Img2Loc: Revisiting Image Geolocalization Using Multi-Modality Foundation Models and Image-Based Retrieval-Augmented Generation](https://dl.acm.org/doi/10.1145/3626772.3657673) ![](https://img.shields.io/badge/date-2024.07-red) 
- [Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs](https://openaccess.thecvf.com/content/CVPR2024/html/Caffagni_Wiki-LLaVA_Hierarchical_Retrieval-Augmented_Generation_for_Multimodal_LLMs_CVPR_2024_paper.html) ![](https://img.shields.io/badge/date-2024.05-red)

#### 🎡 Adaptive and Iterative Retrieval
- [Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent](https://arxiv.org/abs/2411.02937) ![](https://img.shields.io/badge/date-2025.05-red)
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085) ![](https://img.shields.io/badge/date-2025.03-red)
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://api.semanticscholar.org/CorpusID:274192536) ![](https://img.shields.io/badge/date-2024.11-red)
- [RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models](https://aclanthology.org/2024.fever-1.29/) ![](https://img.shields.io/badge/date-2024.11-red)
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/) ![](https://img.shields.io/badge/date-2024.11-red)
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321) ![](https://img.shields.io/badge/date-2024.10-red)
- [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1) ![](https://img.shields.io/badge/date-2024.10-red)
- [Enhancing Multi-modal Multi-hop Question Answering via Structured Knowledge and Unified Retrieval-Generation](https://doi.org/10.1145/3581783.3611964) ![](https://img.shields.io/badge/date-2023.10-red)


### 🤖 Generation Techniques
#### 🧠 In-Context Learning 
- [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red)
- [How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning? (MSIER)](https://aclanthology.org/2024.emnlp-main.305/) ![](https://img.shields.io/badge/date-2024.11-red)
- [RAVEN: Multitask Retrieval Augmented Vision-Language Learning](https://arxiv.org/abs/2406.19150) ![](https://img.shields.io/badge/date-2024.06-red)
- [Retrieval Meets Reasoning: Even High-school Textbook Knowledge Benefits Multimodal Reasoning](https://arxiv.org/abs/2405.20834) ![](https://img.shields.io/badge/date-2024.05-red)
- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828) ![](https://img.shields.io/badge/date-2024.05-red)
- [Retrieval-Augmented Multimodal Language Modeling (RA-CM3)](https://proceedings.mlr.press/v202/yasunaga23a.html) ![](https://img.shields.io/badge/date-2023.06-red)

#### 👨‍⚖️ Reasoning 
- [VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2412.10704) ![](https://img.shields.io/badge/date-2025.02-red)
- [RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models](https://aclanthology.org/2024.fever-1.29/) ![](https://img.shields.io/badge/date-2024.11-red)
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://paperswithcode.com/paper/self-adaptive-multimodal-retrieval-augmented) ![](https://img.shields.io/badge/date-2024.10-red)
- [LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657740) ![](https://img.shields.io/badge/date-2024.07-red)

#### 🤺 Instruction Tuning 
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085) ![](https://img.shields.io/badge/date-2025.03-red)
- [Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning](https://arxiv.org/abs/2501.01120v1) ![](https://img.shields.io/badge/date-2025.01-red)
- [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](https://arxiv.org/abs/2412.14475) ![](https://img.shields.io/badge/date-2024.12-red)
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://arxiv.org/html/2411.15041) ![](https://img.shields.io/badge/date-2024.11-red) 
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154) ![](https://img.shields.io/badge/date-2024.10-red)
- [Rule: Reliable multimodal rag for factuality in medical vision language models](https://arxiv.org/abs/2407.05131) ![](https://img.shields.io/badge/date-2024.10-red)
- [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training (RagVL)](https://arxiv.org/abs/2407.21439) ![](https://img.shields.io/badge/date-2024.09-red)
- [SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information](https://arxiv.org/abs/2409.14083) ![](https://img.shields.io/badge/date-2024.09-red)
- [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://arxiv.org/abs/2404.15516) ![](https://img.shields.io/badge/date-2024.04-red)
- [InstructBLIP: towards general-purpose vision-language models with instruction tuning](https://dl.acm.org/doi/10.5555/3666122.3668264) ![](https://img.shields.io/badge/date-2023.12-red)

#### 📂 Source Attribution and Evidence Transparency 
- [MuRAR: A Simple and Effective Multimodal Retrieval and Answer Refinement Framework for Multimodal Question Answering](https://arxiv.org/abs/2408.08521) ![](https://img.shields.io/badge/date-2025.02-red)
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/abs/2412.14457) ![](https://img.shields.io/badge/date-2024.12-red) 
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75.pdf) ![](https://img.shields.io/badge/date-2024.11-red)


#### 📂 Agentic Generation and Interaction 
- [AppAgent v2: Advanced Agent for Flexible Mobile Interactions](https://arxiv.org/abs/2408.11824) ![](https://img.shields.io/badge/date-2024.10-red)
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/abs/2412.14457) ![](https://img.shields.io/badge/date-2024.12-red) 
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75.pdf) ![](https://img.shields.io/badge/date-2024.11-red)

### 🔧 Training Strategies and Loss Function
- [Improving Medical Multi-modal Contrastive Learning with Expert Annotations](https://doi.org/10.1007/978-3-031-72661-3_27) ![](https://img.shields.io/badge/date-2024.11-red)
- [EchoSight: Advancing Visual-Language Models with Wiki Knowledge](https://aclanthology.org/2024.findings-emnlp.83/) ![](https://img.shields.io/badge/date-2024.11-red)
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103) ![](https://img.shields.io/badge/date-2024.05-red)  
- [HACL: Hallucination Augmented Contrastive Learning for Multimodal Large Language Model](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Hallucination_Augmented_Contrastive_Learning_for_Multimodal_Large_Language_Model_CVPR_2024_paper.html) ![](https://img.shields.io/badge/date-2024.02-red)  
- [Multimodal Learned Sparse Retrieval with Probabilistic Expansion Control](https://arxiv.org/abs/2402.17535) ![](https://img.shields.io/badge/date-2024.02-red)
- [REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.html) ![](https://img.shields.io/badge/date-2023.04-red)


### 🛡️ Robustness and Noise Management
- [AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles](https://arxiv.org/abs/2412.16701) ![](https://img.shields.io/badge/date-2024.12-red)
- [Quantifying the Gaps Between Translation and Native Perception in Training for Multimodal, Multilingual Retrieval](https://aclanthology.org/2024.emnlp-main.335/) ![](https://img.shields.io/badge/date-2024.11-red)
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154) ![](https://img.shields.io/badge/date-2024.10-red)
- [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://aclanthology.org/2024.findings-acl.69/) ![](https://img.shields.io/badge/date-2024.08-red)
- [RAGTrans: Retrieval-Augmented Hypergraph for Multimodal Social Media Popularity Prediction](https://doi.org/10.1145/3637528.3672041) ![](https://img.shields.io/badge/date-2024.08-red)
- [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training (RagVL)](https://arxiv.org/abs/2407.21439) ![](https://img.shields.io/badge/date-2024.07-red)  
- [RA-CM3: Retrieval-Augmented Multimodal Language Modeling](https://arxiv.org/abs/2211.12561) ![](https://img.shields.io/badge/date-2023.01-red)

### 🛠 Tasks Addressed by Multimodal RAGs
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093) ![](https://img.shields.io/badge/date-2024.12-red)  
- [RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2410.14154) ![](https://img.shields.io/badge/date-2024.10-red)  
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321) ![](https://img.shields.io/badge/date-2024.10-red)  
- [Reversed in Time: A Novel Temporal-Emphasized Benchmark for Cross-Modal Video-Text Retrieval](https://dl.acm.org/doi/abs/10.1145/3664647.3680731?casa_token=WcxcaPoWaXIAAAAA:T_4moLyZ5X5J8W137PRysNmdrwx6aH_sMMw1zV9VHeBOGhlJ6rvYqwy3oyzD7Jev3nNkYRlBYbbmJw) ![](https://img.shields.io/badge/date-2024.10-red)  
- [Mllm is a strong reranker: Advancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and noise-injected training](https://arxiv.org/abs/2407.21439) ![](https://img.shields.io/badge/date-2024.09-red)  
- [Ragar, your falsehood radar: Rag-augmented reasoning for political fact-checking using multimodal large language models](https://arxiv.org/abs/2404.12065) ![](https://img.shields.io/badge/date-2024.07-red)  
- [M2-RAAP: A Multi-Modal Recipe for Advancing Adaptation-based Pre-training towards Effective and Efficient Zero-shot Video-text Retrieval](https://dl.acm.org/doi/abs/10.1145/3626772.3657833) ![](https://img.shields.io/badge/date-2024.07-red)  
- [RAVEN: Multitask Retrieval Augmented Vision-Language Learning](https://arxiv.org/abs/2406.19150) ![](https://img.shields.io/badge/date-2024.06-red)  
- [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/abstract/document/10535103/) ![](https://img.shields.io/badge/date-2024.05-red)  
- [A comprehensive survey of hallucination mitigation techniques in large language models](https://www.amanchadha.com/research/2401.01313.pdf) ![](https://img.shields.io/badge/date-2024.01-red)  
- [Ramm: Retrieval-augmented biomedical visual question answering with multi-modal pre-training](https://dl.acm.org/doi/abs/10.1145/3581783.3611830?casa_token=MWu5Fgy31X8AAAAA:C0Fip7NZEfRSiSnzpdf6z9rLnL-kyGkjnN0OdghmCfq7rY0OSPoUGER5jw8_82vFKYE6KArQApEUfA) ![](https://img.shields.io/badge/date-2023.10-red)  
- [Retrieval-augmented multimodal language modeling](https://arxiv.org/abs/2211.12561) ![](https://img.shields.io/badge/date-2023.06-red)  
- [REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.html) ![](https://img.shields.io/badge/date-2023.04-red)  
- [Re-imagen: Retrieval-augmented text-to-image generator](https://arxiv.org/abs/2209.14491) ![](https://img.shields.io/badge/date-2022.11-red)  


#### 🩺 Healthcare and Medicine
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085) ![](https://img.shields.io/badge/date-2025.03-red)  
- [Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation](https://arxiv.org/abs/2407.15268) ![](https://img.shields.io/badge/date-2025.02-red)  
- [Hybrid RAG-Empowered Multi-Modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-Based Contract Approach](https://ieeexplore.ieee.org/abstract/document/10812735?casa_token=LNKXIPFMjI0AAAAA:IcPhMEQM2oJXUbl5beryVYfNp64gFZIVD6kl4bmZHq0rzX1dzDb03xyVR-HbfaxP-IM5aJlshQ) ![](https://img.shields.io/badge/date-2024.12-red)  
- [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://aclanthology.org/2024.emnlp-main.62/) ![](https://img.shields.io/badge/date-2024.11-red)  
- [AsthmaBot: Multi-modal, Multi-Lingual Retrieval Augmented Generation For Asthma Patient Support](https://arxiv.org/abs/2409.15815) ![](https://img.shields.io/badge/date-2024.09-red)  
- [REALM: RAG-Driven Enhancement of Multimodal Electronic Health Records Analysis via Large Language Models](https://arxiv.org/abs/2402.07016) ![](https://img.shields.io/badge/date-2024.02-red)

#### 💻 Software Engineering
- [Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning](https://dl.acm.org/doi/10.1109/icse48619.2023.00205) ![](https://img.shields.io/badge/date-2023.07-red)  
- [Docprompting: Generating code by retrieving the docs](https://drive.google.com/file/d/1v4nEOr4D5z7zpi1nXSONFAqJNeqvitwb/view) ![](https://img.shields.io/badge/date-2023.02-red)  
- [RACE: Retrieval-augmented commit message generation](https://arxiv.org/abs/2203.02700) ![](https://img.shields.io/badge/date-2022.10-red)  
- [Retrieval Augmented Code Generation and Summarization](https://arxiv.org/abs/2108.11601) ![](https://img.shields.io/badge/date-2021.09-red)  

#### 🕶️ Fashion and E-Commerce
- [Unifashion: A unified vision-language model for multimodal fashion retrieval and generation](https://arxiv.org/abs/2408.11305) ![](https://img.shields.io/badge/date-2024.10-red)  
- [Multi-modal Retrieval Augmented Generation for Product Query](https://openurl.ebsco.com/openurl?sid=ebsco:plink:scholar&id=ebsco:gcd:180918461&crl=c) ![](https://img.shields.io/badge/date-2024.07-red)  
- [LLM4DESIGN: An Automated Multi-Modal System for Architectural and Environmental Design](https://arxiv.org/abs/2407.12025) ![](https://img.shields.io/badge/date-2024.06-red)  

#### 🤹 Entertainment and Social Computing
- [SoccerRAG: Multimodal Soccer Information Retrieval via Natural Queries](https://ieeexplore.ieee.org/abstract/document/10859209?casa_token=seYROrW0u9oAAAAA:jclFZaX04YKUh3wKUzBZwKYDgQi_kbUPpFyJPO0HWSgiiqlF3bS9aagz3fUTc0dc9VihYJSiNQ) ![](https://img.shields.io/badge/date-2025.02-red)  
- [Predicting Micro-video Popularity via Multi-modal Retrieval Augmentation](https://dl.acm.org/doi/abs/10.1145/3626772.3657929) ![](https://img.shields.io/badge/date-2024.07-red)  

#### 🚗 Emerging Applications
- [ENWAR: A RAG-empowered Multi-Modal LLM Framework for Wireless Environment Perception](https://arxiv.org/abs/2410.18104) ![](https://img.shields.io/badge/date-2024.10-red)
- [Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications](https://arxiv.org/abs/2410.21943) ![](https://img.shields.io/badge/date-2024.10-red)  
- [Img2Loc: Revisiting image geolocalization using multi-modality foundation models and image-based retrieval-augmented generation](https://dl.acm.org/doi/abs/10.1145/3626772.3657673) ![](https://img.shields.io/badge/date-2024.07-red)
- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828) ![](https://img.shields.io/badge/date-2024.05-red)  

### 📏 Evaluation Metrics
#### 📊 Retrieval Performance
- **Recall@K, Precision@K, F1 Score, and MRR**:  
  - [OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](https://arxiv.org/abs/2412.02592) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [EMERGE: Integrating RAG for Improved Multimodal EHR Predictive Modeling](https://arxiv.org/abs/2406.00036) ![](https://img.shields.io/badge/date-2025.02-red)  
  - [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](https://aclanthology.org/2024.emnlp-main.89/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [Large Language Models Know What is Key Visual Entity: An LLM-assisted Multimodal Retrieval for VQA](https://aclanthology.org/2024.emnlp-main.613/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models](https://aclanthology.org/2024.fever-1.29/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [Rule: Reliable multimodal rag for factuality in medical vision language models](https://arxiv.org/abs/2407.05131) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training (RagVL)](https://arxiv.org/abs/2407.21439) ![](https://img.shields.io/badge/date-2024.09-red)  
  - [M2-RAAP: A Multi-Modal Recipe for Advancing Adaptation-based Pre-training towards Effective and Efficient Zero-shot Video-text Retrieval](https://doi.org/10.1145/3626772.3657833) ![](https://img.shields.io/badge/date-2024.07-red)  
  - [Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657727) ![](https://img.shields.io/badge/date-2024.07-red)  
  - [Retrieval-augmented egocentric video captioning](https://arxiv.org/abs/2401.00789) ![](https://img.shields.io/badge/date-2024.06-red)  
  - [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103) ![](https://img.shields.io/badge/date-2024.05-red)  
  - [MV-Adapter: Multimodal Video Transfer Learning for Video Text Retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Jin_MV-Adapter_Multimodal_Video_Transfer_Learning_for_Video_Text_Retrieval_CVPR_2024_paper.pdf) ![](https://img.shields.io/badge/date-2024.04-red)  
  - [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://arxiv.org/abs/2404.15516) ![](https://img.shields.io/badge/date-2024.04-red)  
  - [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610) ![](https://img.shields.io/badge/date-2024.04-red)  
  - [Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval](https://arxiv.org/abs/2403.17998) ![](https://img.shields.io/badge/date-2024.03-red)  
  - [REALM: RAG-Driven Enhancement of Multimodal Electronic Health Records Analysis via Large Language Models](https://arxiv.org/abs/2402.07016) ![](https://img.shields.io/badge/date-2024.02-red)  
  - [Multimodal Learned Sparse Retrieval with Probabilistic Expansion Control](https://arxiv.org/abs/2402.17535) ![](https://img.shields.io/badge/date-2024.02-red)  
  - [VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering](https://arxiv.org/abs/2312.12273) ![](https://img.shields.io/badge/date-2023.12-red)  
  - [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928) ![](https://img.shields.io/badge/date-2022.10-red)  

- **min(+P, Se)**:

It represents the minimum value between precision (+P) and sensitivity (Se), providing a balanced measure of model performance.
  - [EMERGE: Integrating RAG for Improved Multimodal EHR Predictive Modeling](https://arxiv.org/abs/2406.00036) ![](https://img.shields.io/badge/date-2025.02-red)  
  - [REALM: RAG-Driven Enhancement of Multimodal Electronic Health Records Analysis via Large Language Models](https://arxiv.org/abs/2402.07016) ![](https://img.shields.io/badge/date-2024.02-red)  

#### 📝 Fluency and Readability
- **Fluency (FL)**:  
  - [Multi-modal Retrieval Augmented Multi-modal Generation: A Benchmark, Evaluate Metrics and Strong Baselines](https://arxiv.org/abs/2411.16365) ![](https://img.shields.io/badge/date-2025.05-red) 
  - [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red) 
  - [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928) ![](https://img.shields.io/badge/date-2022.10-red) 


#### ✅ Relevance and Accuracy
- **Accuracy**:  
  - [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [MRAG-BENCH: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models](https://arxiv.org/abs/2410.08182) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [Enhancing Textbook Question Answering Task with Large Language Models and Retrieval Augmented Generation](https://arxiv.org/abs/2402.05128) ![](https://img.shields.io/badge/date-2025.01-red)  
  - [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://api.semanticscholar.org/CorpusID:274192536) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?](https://aclanthology.org/2024.emnlp-main.305/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://arxiv.org/abs/2407.05131) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-Augmented Generation via Knowledge-Enhanced Reranking and Noise-Injected Training (RagVL)](https://arxiv.org/abs/2407.21439) ![](https://img.shields.io/badge/date-2024.09-red)  
  - [Advanced Embedding Techniques in Multimodal Retrieval Augmented Generation: A Comprehensive Study on Cross Modal AI Applications](https://drpress.org/ojs/index.php/jceim/article/view/24094) ![](https://img.shields.io/badge/date-2024.07-red) 
  - [RAVEN: Multitask Retrieval Augmented Vision-Language Learning](https://arxiv.org/abs/2406.19150) ![](https://img.shields.io/badge/date-2024.06-red)  
  - [Retrieval Meets Reasoning: Even High-School Textbook Knowledge Benefits Multimodal Reasoning](https://arxiv.org/abs/2405.20834) ![](https://img.shields.io/badge/date-2024.05-red)  
  - [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928) ![](https://img.shields.io/badge/date-2022.10-red)  

#### 🖼️ Image-related Metrics
- **Fréchet Inception Distance (FID), CLIP Score, Kernel Inception Distance (KID), and Inception Score (IS)**:
  - [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](https://aclanthology.org/2024.emnlp-main.89/) ![](https://img.shields.io/badge/date-2024.11-red)
  - [C3Net: Compound Conditioned ControlNet for Multimodal Content Generation](https://arxiv.org/abs/2311.17951) ![](https://img.shields.io/badge/date-2023.11-red) 
  - [Retrieval-Augmented Multimodal Language Modeling (RA-CM3)](https://proceedings.mlr.press/v202/yasunaga23a.html) ![](https://img.shields.io/badge/date-2023.06-red) 

- **Consensus-Based Image Description Evaluation (CIDEr)**:
  - [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/abs/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [RAVEN: Multitask Retrieval Augmented Vision-Language Learning](https://arxiv.org/abs/2406.19150) ![](https://img.shields.io/badge/date-2024.06-red)  
  - [Retrieval-augmented egocentric video captioning](https://arxiv.org/abs/2401.00789) ![](https://img.shields.io/badge/date-2024.06-red)  
  - [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828) ![](https://img.shields.io/badge/date-2024.05-red)  
  - [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610) ![](https://img.shields.io/badge/date-2024.04-red)  
  - [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](https://aclanthology.org/2024.emnlp-main.89/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [MSIER: How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?](https://aclanthology.org/2024.emnlp-main.305/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [C3Net: Compound Conditioned ControlNet for Multimodal Content Generation](https://arxiv.org/abs/2311.17951) ![](https://img.shields.io/badge/date-2023.11-red)  
  - [Retrieval-Augmented Multimodal Language Modeling (RA-CM3)](https://proceedings.mlr.press/v202/yasunaga23a.html) ![](https://img.shields.io/badge/date-2023.06-red)  
  - [REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.html) ![](https://img.shields.io/badge/date-2023.04-red)  

- **SPICE**:
  - [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/abs/2405.10311) ![](https://img.shields.io/badge/date-2025.04-red)  

- **SPIDEr**:
  - [C3Net: Compound Conditioned ControlNet for Multimodal Content Generation](https://arxiv.org/abs/2311.17951) ![](https://img.shields.io/badge/date-2023.11-red) 

 
#### 🎵 Audio-related Metrics
- **Fréchet Audio Distance (FAD), Overall Quality (OVL), and Text Relevenace (REL)**:  
  - [C3Net: Compound Conditioned ControlNet for Multimodal Content Generation](https://arxiv.org/abs/2311.17951) ![](https://img.shields.io/badge/date-2023.11-red) 
  - [AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352) ![](https://img.shields.io/badge/date-2023.03-red)

#### 🔗 Text Similarity and Overlap Metrics
- **BLEU, METEOR, and ROUGE-L**:  

  - [Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation](https://arxiv.org/abs/2407.15268) ![](https://img.shields.io/badge/date-2025.02-red)  
  - [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/pdf/2405.10311) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation](https://aclanthology.org/2024.emnlp-main.89/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags](https://aclanthology.org/2024.findings-acl.771/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [Rule: Reliable multimodal rag for factuality in medical vision language models](https://arxiv.org/abs/2407.05131) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [AsthmaBot: Multi-modal, Multi-Lingual Retrieval Augmented Generation For Asthma Patient Support](https://arxiv.org/abs/2409.15815) ![](https://img.shields.io/badge/date-2024.09-red)  
  - [Advanced Embedding Techniques in Multimodal Retrieval Augmented Generation: A Comprehensive Study on Cross Modal AI Applications](https://drpress.org/ojs/index.php/jceim/article/view/24094) ![](https://img.shields.io/badge/date-2024.07-red) 
  - [RAVEN: Multitask Retrieval Augmented Vision-Language Learning](https://arxiv.org/abs/2406.19150) ![](https://img.shields.io/badge/date-2024.06-red)  
  - [Retrieval-augmented egocentric video captioning](https://arxiv.org/abs/2401.00789) ![](https://img.shields.io/badge/date-2024.06-red)  
  - [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828) ![](https://img.shields.io/badge/date-2024.05-red)  
  - [Do You Remember? Dense Video Captioning with Cross-Modal Memory Retrieval](https://arxiv.org/abs/2404.07610) ![](https://img.shields.io/badge/date-2024.04-red)  

- **Exact Match (EM)**:
  - [OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](https://arxiv.org/abs/2412.02592) ![](https://img.shields.io/badge/date-2025.03-red)  
  - [Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [Large Language Models Know What is Key Visual Entity: An LLM-assisted Multimodal Retrieval for VQA](https://aclanthology.org/2024.emnlp-main.613/) ![](https://img.shields.io/badge/date-2024.11-red)  
  - [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1) ![](https://img.shields.io/badge/date-2024.10-red)  
  - [MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training (RagVL)](https://arxiv.org/abs/2407.21439) ![](https://img.shields.io/badge/date-2024.09-red)  
  - [UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models](https://ieeexplore.ieee.org/document/10535103) ![](https://img.shields.io/badge/date-2024.05-red)  
  - [MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text](https://arxiv.org/abs/2210.02928) ![](https://img.shields.io/badge/date-2022.10-red)  

- **BERTScore**:
  - [Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation](https://arxiv.org/abs/2407.15268) ![](https://img.shields.io/badge/date-2025.02-red)  
  - [XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags](https://aclanthology.org/2024.findings-acl.771/) ![](https://img.shields.io/badge/date-2024.11-red) 

#### 📊 Statistical Metrics
- **Spearman’s Rank Correlation (SRC)**:  
  - [Predicting Micro-video Popularity via Multi-modal Retrieval Augmentation](https://doi.org/10.1145/3626772.3657929) ![](https://img.shields.io/badge/date-2024.07-red) 

#### ⚙️ Efficiency and Computational Performance
- **Average Retrieval Time per Query**:  
  - [Advanced Embedding Techniques in Multimodal Retrieval Augmented Generation: A Comprehensive Study on Cross Modal AI Applications](https://drpress.org/ojs/index.php/jceim/article/view/24094) ![](https://img.shields.io/badge/date-2024.07-red) 

- **FLOPs (Floating Point Operations)**:  
  - [Multimodal Learned Sparse Retrieval with Probabilistic Expansion Control](https://arxiv.org/abs/2402.17535) ![](https://img.shields.io/badge/date-2024.02-red) 

- **Response Time**:
   - [Multi-modal Retrieval Augmented Generation for Product Query](https://bpasjournals.com/library-science/index.php/journal/article/view/2437) ![](https://img.shields.io/badge/date-2024.07-red) 

- **Execution Time**:
   - [SoccerRAG: Multimodal Soccer Information Retrieval via Natural Queries](https://arxiv.org/abs/2406.01273) ![](https://img.shields.io/badge/date-2024.07-red) 

- **Average Retrieval Number (ARN)**:
   - [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321) ![](https://img.shields.io/badge/date-2024.10-red) 

#### 🏥 Domain-Specific Metrics
- **Clinical Relevance (CR)**:
   - [AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles](https://arxiv.org/abs/2412.16701) ![](https://img.shields.io/badge/date-2025.06-red) 
- **Geodesic Distance**:
  - [Img2Loc: Revisiting Image Geolocalization Using Multi-Modality Foundation Models and Image-Based Retrieval-Augmented Generation](https://dl.acm.org/doi/10.1145/3626772.3657673) ![](https://img.shields.io/badge/date-2024.07-red) 


---

**This README is a work in progress and will be completed soon. Stay tuned for more updates!**

---
## 🔗 Citations
If you find our paper or repository useful, please cite the paper:
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


## ⭐ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=llm-lab-org/Multimodal-RAG-Survey&type=Date)](https://www.star-history.com/#llm-lab-org/Multimodal-RAG-Survey&Date)
