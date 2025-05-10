# Research Project: Generalization of Handwriting OCR Through Dataset Style Variation

This project explores how differences in handwriting style diversity impact model robustness and cross-domain performance, specifically using TrOCR model as base model and Imgur and IAM databases. 

## Installation
Complete the following steps:
1. Clone the repository to your desired location using ```[git clone https://github.com/SeoliKim/stylized-handwriting-ocr.git```.
2. Create python virtual environment using *requirements.txt*
3. Download the following database:
   - **IMGUR database**: [IMGUR5K Handwriting Dataset Repository](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset)
   - **IAM database**: [Kaggle IAM Database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)

## Quick Start
Run the [`code-overview.ipynb`](code-overview.ipynb) to go through an overview the data-processing, training and testing process. 


## Usage
1.  For Imgur dataset, run [`imgur-process_dataset.ipynb`](/imgur/dataset/process_dataset.ipynb) to crop full-page images into word images and save the word-image data on the respective directory.
2.  Make sure  *dataset_info* for IAM and IMGUR dataset are in the same folder with the downloaded dataset.
3.  Run **train_script.py** in `iam/` and `imgur/` to train the model.
4.  Run **eval_script.py** in `iam/` and `imgur/` to evaluate the model.

## Results
The output of our research is stored in [`results`](/results) folder, this includes evaluation on the Imgur and IAM testing set for the following 4 models:
- base-imgur: [microsoft/trocr-base-stage1](https://huggingface.co/microsoft/trocr-base-stage1) fine-tuned on Imgur dataset
- small-imgur: [microsoft/trocr-small-stage1](https://huggingface.co/microsoft/trocr-small-stage1) fine-tuned on Imgur dataset
- base-iam: [microsoft/trocr-base-stage1](https://huggingface.co/microsoft/trocr-base-stage1) fine-tuned on IAM dataset
- small-iam: [microsoft/trocr-small-stage1](https://huggingface.co/microsoft/trocr-small-stage1) fine-tuned on IAM dataset
- [base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten): public fine-tuned model
- [small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten): public fine-tuned model


