# Research Project: Handwritten Text Recognition using TrOCR

This repository contains the code and resources for fine-tuning a TrOCR-based model for handwritten text recognition.

## Methodology

The pipeline consists of the following steps:

For **Stylized IMGUR5K Handwriting Dataset**: 

1.  Obtain the full-page strlyed handwritten text dataset from [IMGUR5K Handwriting Dataset Repository](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) 

2. Run [`process_dataset.ipynb`](/stylized/dataset/process_dataset.ipynb) to crop words from full-page images and save them as individual word images.
   - The processed words are saved as pickle files for efficient loading.

3. The dataset is separated into training and testing sets (typically 80-20 split) for consistent evaluation.
   - Directory: [`save_train_test.ipynb`](/stylized/dataset/save_train_test.ipynb)

4. Use [`train_script`](/stylized/train/train_script.py) to fine-tune the TrOCR-based model on the training dataset.
   - For our research, we fine-tuned on two base models:
      - microsoft/trocr-base-stage1
      - microsoft/trocr-base-stage1

5. Use [`test_script`](/stylized/test/test_script.py) to evaluate the model performance on the test set.


