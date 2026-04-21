# Unveiling the Latent Directions of Reflection in Large Language Models

This repository contains an implementation for the paper **Unveiling the Latent Directions of Reflection in Large Language Models**. [https://arxiv.org/abs/2508.16989](https://arxiv.org/abs/2508.16989)

**TL;DR:** We show that LLM reflection can be categorized into three levels and can be controlled using activation steering.

## Run experiments
If you use gated Hugging Face models, export your token first:
```sh
export HF_TOKEN="<YOUR_TOKEN_HERE>"
```
Setup venv:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```
Run all the experiments:
```sh
bash run_experiments.sh
```
