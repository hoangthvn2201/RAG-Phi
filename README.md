# RAG-Phi: Enhancing CXR report generation task with light-weight Vision Language Models
This project aims to build a Retrieval-Augmented Generation (RAG) system to support lightweight vision-language models in generating medical reports. Instead of changing the model itself, we focus on adding a retrieval module that brings in useful medical information from external sources. The main goal is to improve the accuracy and completeness of the reports while keeping the system fast and easy to run. This makes it more practical for use in real hospitals and clinics.

## Requirements
1. Clone this repository and navigate to RAG-Phi folder
```bash
git clone https://github.com/hoangthvn2201/RAG-Phi.git
cd RAG-Phi
```

2. Install Package: Create conda environment

```Shell
conda create -n RAG-Phi python=3.10 -y
conda activate RAG-Phi
cd RAG-Phi
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install trl
```

3. Download the required model checkpoints [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.