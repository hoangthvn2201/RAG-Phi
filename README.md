# RAG-Phi: Enhancing CXR report generation task with light-weight Vision Language Models
This project aims to build a Retrieval-Augmented Generation (RAG) system to support lightweight vision-language models in generating medical reports. Instead of changing the model itself, we focus on adding a retrieval module that brings in useful medical information from external sources. The main goal is to improve the accuracy and completeness of the reports while keeping the system fast and easy to run. This makes it more practical for use in real hospitals and clinics.

## Methodology 
### 1. Retriever Design
Inspired by [RULE](https://arxiv.org/abs/2407.05131), we designed a retriever that encodes both medical images and textual reports into a shared embedding space. Specifically, the vision encoder processes chest X-ray images, while the text encoder embeds the associated medical reports. During inference, given a target image $x_t$​, the system retrieves the top-K most similar medical reports from the training corpus based on embedding similarity. These retrieved reports act as contextual references to guide the generation of the final report for $x_t$

![System Architecture][system-architecture]
### 2. Calibrated Retrieved Context Selection
A challenge in RAG systems is the trade-off between coverage and relevance. While retrieving too few reports may omit critical information, retrieving too many can introduce noisy or unrelated context. Inspired by the MMed-RAG method [1], we implement an adaptive truncation strategy. Let $S_i$ be the similarity score of the $ith$ retrieved context. We compute a similarity drop ratio:
```math
u_i = \log\left( \frac{S_i}{S_{i+1}} \right), \quad \text{for } 0 < i \leq K
```

If ui ​exceeds a defined threshold , it indicates a sharp drop in relevance, and the retrieval list is truncated at that point. This ensures that only the most relevant contextual data is used in generation, enhancing output quality.

### 3. Domain-Adaptive Retriever Fine-tuning
To enhance the retrieval quality in the chest X-ray domain, we perform domain-adaptive fine-tuning on the retriever using paired image-report data. Specifically, we adopt the pretrained model ***[microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)*** as the base retriever.

## ⚙️ Setup
### 1. Clone this repository and navigate to RAG-Phi folder
```bash
git clone https://github.com/hoangthvn2201/RAG-Phi.git
cd RAG-Phi
```

### 2. Install Package: Create conda environment

```Shell
conda create -n RAG-Phi python=3.10 -y
conda activate RAG-Phi
cd RAG-Phi
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install trl
```

### 3. Download the required model checkpoints [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.

## 📞 Contact

For questions, feedback, or contributions, please reach out via:

- 📧 **Email**: [huyhoangt2201@gmail.com](mailto:huyhoangt2201@gmail.com)  
- 🌐 **LinkedIn**: [@huyhoangt2004](https://www.linkedin.com/in/huyhoangt2004/) 
<p align="right">(<a href="#readme-top">back to top</a>)</p>
[system-architecture]: images/system_architecture.png
