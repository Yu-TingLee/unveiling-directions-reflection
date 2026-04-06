# Unveiling the Latent Directions of Reflection in Large Language Models

This repository contains an implementation for the paper **Unveiling the Latent Directions of Reflection in Large Language Models**. [https://arxiv.org/abs/2508.16989](https://arxiv.org/abs/2508.16989)


If you use gated Hugging Face models, export your token first:
```sh
export HF_TOKEN="<YOUR_TOKEN_HERE>"
```
## Setup venv

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run experiments


```sh
python setup_models.py
bash run_experiments.sh
```
