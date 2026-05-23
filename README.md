# Unveiling the Latent Directions of Reflection in Large Language Models

This repository contains an implementation for the paper: **Unveiling the Latent Directions of Reflection in Large Language Models**, available on [ArXiv](https://arxiv.org/abs/2508.16989).

## Short Abstract
**TL;DR:** We show that LLM reflection can be categorized into three levels and can be controlled using activation steering.

While reflection enables language models to evaluate and revise reasoning, its inner mechanisms remain underexplored. We investigate reflection through the lens of latent directions in model activations. By constructing steering vectors across different reflection levels (none, intrinsic, and triggered), we demonstrate that reflective behavior can be directly enhanced or suppressed through activation interventions. Experiments on GSM8k-adv and Cruxeval-o-adv confirm this controllability, revealing that suppressing reflection is considerably easier than stimulating it. These findings highlight both defensive opportunities and adversarial risks, opening a path toward a mechanistic understanding of reflective reasoning in LLMs.


## Usage
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
