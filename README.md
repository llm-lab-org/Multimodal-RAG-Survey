# From Text to Multimodality: A Survey on Multimodal Retrieval-Augmented Generation

This repository is designed to collect and categorize papers related to Multimodal Retrieval-Augmented Generation (RAG) according to our survey paper: [Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2502.08826). Given the rapid growth in this field, we will continuously update both the paper and this repository to serve as a resource for researchers working on future projects.

## General Pipeline of Multimodal RAG
![output-onlinepngtools (1)](https://github.com/user-attachments/assets/fabab7c0-9ca3-4d0b-b4d5-fb46defc8620)

## Taxonomy of recent advances and enhancements in multimodal RAG
![6634_Ask_in_Any_Modality_A_Com_organized-1-cropped](https://github.com/user-attachments/assets/0b5cd8e6-1aef-402b-a0a3-e3bf5cf555ae)

## Taxonomy of application domains for multimodal RAG.
![6634_Ask_in_Any_Modality_A_Com_organized-2-cropped](https://github.com/user-attachments/assets/f46ac78b-f51a-43c4-90bc-938d441093f2)


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

### Audio-Text

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| LibriSpeech     | 1,000 hours of read English speech with corresponding text; ASR corpus based on audiobooks.                     | Audio, Text      | [LibriSpeech](https://www.openslr.org/12)                                                           |
| SpeechBrown     | 55K paired speech-text samples; 15 categories covering diverse topics from religion to fiction.                 | Audio, Text      | [SpeechBrown](https://huggingface.co/datasets/llm-lab/SpeechBrown)                                   |
| AudioCap        | 46K audio clips paired with human-written text captions.                                                       | Audio, Text      | [AudioCaps](https://audiocaps.github.io/)                                                           |
| AudioSet        | 2M human-labeled sound clips from YouTube across diverse audio event classes (e.g., music or environmental).     | Audio            | [AudioSet](https://research.google.com/audioset/)                                                   |

---

### Medical

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| MIMIC-CXR       | 125,417 labeled chest X-rays with reports; widely used for medical imaging research.                            | Image, Text      | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)                                         |
| CheXpert        | 224,316 chest radiographs of 65,240 patients; focused on medical analysis.                                      | Image, Text      | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)                               |
| MIMIC-III       | Health-related data from over 40K patients; includes clinical notes and structured data.                        | Text             | [MIMIC-III](https://mimic.physionet.org/)                                                           |
| IU-Xray         | 7,470 pairs of chest X-rays and corresponding diagnostic reports.                                               | Image, Text      | [IU-Xray](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)                   |
| PubLayNet       | 100,000 training samples and 2,160 test samples built from PubLayNet for document layout analysis.              | Image, Text      | [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)                                               |

---

### Fashion

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| Fashion-IQ       | 77,684 images across three categories; evaluated with Recall@10 and Recall@50 metrics.                         | Image, Text      | [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq)                                             |
| FashionGen       | 260.5K image–text pairs of fashion images and item descriptions.                                               | Image, Text      | [FashionGen](https://www.elementai.com/datasets/fashiongen)                                         |
| VITON-HD         | 83K images for virtual try-on; high-resolution clothing items dataset.                                         | Image, Text      | [VITON-HD](https://github.com/shadow2496/VITON-HD)                                                  |
| Fashionpedia     | 48,000 fashion images annotated with segmentation masks and fine-grained attributes.                           | Image, Text      | [Fashionpedia](https://fashionpedia.ai/)                                                            |
| DeepFashion      | Approximately 800K diverse fashion images for pseudo triplet generation.                                       | Image, Text      | [DeepFashion](https://github.com/zalandoresearch/fashion-mnist)                                     |

---

### Knowledge & QA

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

### Other

| **Name**         | **Statistics and Description**                                                                                  | **Modalities**   | **Link**                                                                                             |
|------------------|------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------|
| ImageNet         | 14M labeled images across thousands of categories; used as a benchmark in computer vision research.             | Image            | [ImageNet](http://www.image-net.org/)                                                               |
| Oxford Flowers102| Dataset of flowers with 102 categories for fine-grained image classification tasks.                            | Image            | [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)                            |
| Stanford Cars    | Images of different car models (five examples per model); used for fine-grained categorization tasks.           | Image            | [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)                |
| GeoDE            | 61,940 images from 40 classes across six world regions; emphasizes geographic diversity in object recognition.   | Image            | [GeoDE](https://github.com/AliRamazani/GeoDE)                                                       |

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
