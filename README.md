# Research Project: Generalization of Handwriting OCR Through Dataset Style Variation

This project explores how differences in handwriting style diversity impact model robustness and cross-domain performance, specifically using TrOCR model as base model and Imgur and IAM databases. 

## Installation
Complete the following steps:
1. Clone the repository to your desired location using ```git clone https:///SeoliKim/stylized-handwriting-ocr```.
2. Create python virtual environment using *requirements.txt*
3. Download the following database:
   - **IMGUR database**: [IMGUR5K Handwriting Dataset Repository](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset)
   - **IAM database**: [Kaggle IAM Database]([https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database))

## Usage
1.  For Imgur dataset, run [`imgur-process_dataset.ipynb`](/imgur/dataset/process_dataset.ipynb) to crop full-page images into word images and save the word-image data on the respective directory.
2.  Move  

3. Run [`process_dataset.ipynb`](/stylized/dataset/process_dataset.ipynb) to crop words from full-page images and save them as individual word images.
   - The processed words are saved as pickle files for efficient loading.

4. The dataset is separated into training and testing sets (typically 80-20 split) for consistent evaluation.
   - Directory: [`save_train_test.ipynb`](/stylized/dataset/save_train_test.ipynb)

5. Use [`train_script`](/stylized/train/train_script.py) to fine-tune the TrOCR-based model on the training dataset.
   - For our research, we fine-tuned on two base models:
      - microsoft/trocr-base-stage1
      - microsoft/trocr-base-stage1

6. Use [`test_script`](/stylized/test/test_script.py) to evaluate the model performance on the test set.


