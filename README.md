# From Text to Multimodality: A Survey on Multimodal Retrieval-Augmented Generation

This repository is designed to collect and categorize papers related to Multimodal Retrieval-Augmented Generation (RAG) according to our survey paper: [From Text to Multimodality: A Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2502.08826). Given the rapid growth in this field, we will continuously update both the paper and this repository to serve as a resource for researchers working on future projects.

## General Pipeline of Multimodal RAG
![output-onlinepngtools (1)](https://github.com/user-attachments/assets/fabab7c0-9ca3-4d0b-b4d5-fb46defc8620)

## Taxonomy of recent advances and enhancements in multimodal RAG
![Multimodal_Retrieval_Augmented_Generation__A_Survey__V2__organized (1) (cropped) (pdfresizer com) (1)_page-0001](https://github.com/user-attachments/assets/987874e8-bfa5-4563-8949-676d17cdaedb)


## Abstract
Large Language Models (LLMs) struggle with hallucinations and outdated knowledge due to their reliance on static training data. Retrieval-Augmented Generation (RAG) mitigates these issues by integrating external dynamic information enhancing factual and updated grounding. Recent advances in multimodal learning have led to the development of Multimodal RAG, incorporating multiple modalities such as text, images, audio, and video to enhance the generated outputs. However, cross-modal alignment and reasoning introduce unique challenges to Multimodal RAG, distinguishing it from traditional unimodal RAG.

This survey offers a structured and comprehensive analysis of Multimodal RAG systems, covering datasets, metrics, benchmarks, evaluation, methodologies, and innovations in retrieval, fusion, augmentation, and generation. We precisely review training strategies, robustness enhancements, and loss functions, while also exploring the diverse Multimodal RAG scenarios. 
Furthermore, we discuss open challenges and future research directions to support advancements in this evolving field. This survey lays the foundation for developing more capable and reliable AI systems that effectively leverage multimodal dynamic external knowledge bases. 

## Citations
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

## Overview of Popular Datasets in Multimodal RAG Research

### Image-Text General

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

### Video-Text

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

---

### Audio-Text

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| LibriSpeech     | 1,000 hours of read English speech with corresponding text; ASR corpus based on audiobooks.                     | Audio, Text      | [LibriSpeech](https://www.openslr.org/12)                                                           |
| AudioCap        | 46K audio clips paired with human-written text captions.                                                       | Audio, Text      | [AudioCaps](https://audiocaps.github.io/)                                                           |
| AudioSet        | 2M human-labeled sound clips from YouTube across diverse audio event classes (e.g., music or environmental).     | Audio            | [AudioSet](https://research.google.com/audioset/)                                                   |

---

## Paper Collection
### RAG-related Surveys
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


### Retrieval Strategies Advances
#### Maximum Inner Product Search (MIPS)

### Augmentation Technique
#### Context-Enrichment 

- [EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation](https://doi.org/10.1145/3627673.3679582)  

- [Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering](https://aclanthology.org/2024.emnlp-main.922/)  
- [Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs](https://openaccess.thecvf.com/content/CVPR2024/html/Caffagni_Wiki-LLaVA_Hierarchical_Retrieval-Augmented_Generation_for_Multimodal_LLMs_CVPR_2024_paper.html)  
- [Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](https://arxiv.org/abs/2411.13093)  
- [Img2Loc: Revisiting Image Geolocalization Using Multi-Modality Foundation Models and Image-Based Retrieval-Augmented Generation](https://doi.org/10.1145/3627673.3679582)  
- [Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering](https://arxiv.org/abs/2412.20927) 

#### Adaptive and Iterative Retrieval

- [Enhancing Multi-modal Multi-hop Question Answering via Structured Knowledge and Unified Retrieval-Generation](https://doi.org/10.1145/3581783.3611964)  
- [Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation](http://dx.doi.org/10.36227/techrxiv.172840252.24352951/v1)  
- [OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems](https://aclanthology.org/2024.emnlp-industry.75/)  
- [Self-adaptive Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2410.11321)  
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085)  
- [Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent](https://arxiv.org/abs/2411.02937)  
- [mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA](https://api.semanticscholar.org/CorpusID:274192536)  
- [RAGAR, Your Falsehood Radar: RAG-Augmented Reasoning for Political Fact-Checking using Multimodal Large Language Models](https://aclanthology.org/2024.fever-1.29/)

---
## Contact
If you have questions, please send an email to mahdi.abootorabi2@gmail.com.
