#!/usr/bin/env python
# coding: utf-8

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TROCR Model Evaluation")
    parser.add_argument(
        "--project-dir",
        type=str,
        default='/common/users/$id/project',
        help="Path where the project root is"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default='dataset',
        help="Path where the trained dataset is stored"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Name/path of the model to use"
    )
    parser.add_argument(
        "--processor-name",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Name/path of the processor to use"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/train",
        help="Path to save trained model"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Now use args.model_name, args.processor_name, args.result_path in your code
    print(f"Using model: {args.model_name}")
    print(f"Using processor: {args.processor_name}")
    print(f"Model will be saved to: {args.model_dir}")

    # Step 1. Prepare Dataset

    # 1.1 Loading the Image Data
    import os
    
    # Change to your desired directory
    os.chdir(args.project_dir)
    # Confirm it's changed
    print("Current directory:", os.getcwd())

    # load all data
    import pickle
    
    with open(args.dataset_dir+'/dfwords_0_20000.pkl', 'rb') as file:
        loaded_dfwords1 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_20000_40000.pkl', 'rb') as file:
        loaded_dfwords2 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_40000_60000.pkl', 'rb') as file:
        loaded_dfwords3 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_60000_80000.pkl', 'rb') as file:
        loaded_dfwords4 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_80000_100000.pkl', 'rb') as file:
        loaded_dfwords5 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_100000_120000.pkl', 'rb') as file:
        loaded_dfwords6 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_120000_140000.pkl', 'rb') as file:
        loaded_dfwords7 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_140000_160000.pkl', 'rb') as file:
        loaded_dfwords8 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_160000_180000.pkl', 'rb') as file:
        loaded_dfwords9 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_180000_200000.pkl', 'rb') as file:
        loaded_dfwords10 = pickle.load(file)
    with open(args.dataset_dir+'/dfwords_200000_227055.pkl', 'rb') as file:
        loaded_dfwords11 = pickle.load(file)

    # merge into one dataframe
    import pandas as pd
    
    loaded_dfwords = pd.concat([loaded_dfwords1, loaded_dfwords2], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords3], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords4], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords5], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords6], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords7], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords8], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords9], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords10], ignore_index=True)  # vertical stack
    loaded_dfwords = pd.concat([loaded_dfwords, loaded_dfwords11], ignore_index=True)  # vertical stack

    # confirm if loaded correctly
    loaded_dfwords.info()
    loaded_dfwords.head(3)


    # 1.2 Loading the Training Data
    df_train_info= pd.read_csv(args.dataset_dir+"/dataset_info/df_train_info.csv") 
    df_train_info.head()
    df_train_info_premerge= df_train_info.drop(['id', 'text'], axis=1)

    df_train= pd.merge(
        loaded_dfwords.reset_index(), 
        df_train_info_premerge, 
        left_on='index', 
        right_on='word_idx',
        how='right'  
    )

    # check if df_train is loaded correctly
    df_train.info()
    df_train.head()


    # 1.3 Splitting the Training Data into Training and Validation Subsets
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    train_df, eval_df = train_test_split(df_train, test_size=0.2, random_state=42)
    
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    # get sample of 8000, match 20 % of training reduce
    eval_df = eval_df.sample(n=8000, random_state=42).reset_index()
    
    train_df.info()
    eval_df.info()

    # Step 2. Running the Model

    # 2.1 Loading the Model
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    # get base model
    processor = TrOCRProcessor.from_pretrained(args.processor_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)


    # 2.2 Process Dataset with Processor

    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    
    class StyleDataset(Dataset):
        def __init__(self, df, processor, max_target_length=512):
            self.df = df
            self.processor = processor
            self.max_target_length = max_target_length
    
        def __len__(self):
            return len(self.df)
    
        def __getitem__(self, idx):
          try:
              text = self.df['text'][idx]
              if not isinstance(text, str) or not text.strip():
                  raise ValueError(f"Invalid text at index {idx}: {repr(text)}")
              image_id = self.df['id'][idx]
              try:
                  image = self.df['image'][idx]
              except Exception as e:
                  raise ValueError(f"Failed to load image for ID {image_id} at index {idx}") from e
              try:
                  pixel_values = self.processor(image, return_tensors="pt").pixel_values
              except Exception as e:
                  raise ValueError(f"Image processing failed at index {idx}") from e
    
              if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                  raise ValueError(f"Invalid pixel values (NaN/inf) at index {idx}")
              try:
                  labels = self.processor.tokenizer(
                      text,
                      padding="max_length",
                      max_length=self.max_target_length
                  ).input_ids
              except Exception as e:
                  raise ValueError(f"Tokenization failed for text at index {idx}") from e
    
              # Replace pad_token_id with -100 for loss masking
              labels = [
                  label if label != self.processor.tokenizer.pad_token_id else -100
                  for label in labels
              ]
              encoding = {
                  "pixel_values": pixel_values.squeeze(),
                  "labels": torch.tensor(labels)
              }
    
              if encoding["pixel_values"].dim() != 3:
                  raise ValueError(f"Invalid pixel_values shape at index {idx}")
    
              if encoding["labels"].numel() != self.max_target_length:
                  raise ValueError(f"Labels length mismatch at index {idx}")
    
              return encoding
    
          except Exception as e:
              print(f"\nError in sample {idx}:")
              print(f"   Error type: {type(e).__name__}")
              print(f"   Details: {str(e)}")
              if hasattr(e, '__cause__') and e.__cause__:
                  print(f"   Underlying error: {type(e.__cause__).__name__}: {str(e.__cause__)}")
              print(f"   DataFrame row:\n{self.df.iloc[idx]}")
              return None

    # Process train and eval dataset
    train_dataset = StyleDataset(df=train_df,processor=processor)
    eval_dataset= StyleDataset(df=eval_df,processor=processor)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    # get the label string from encoding
    def get_label_str(encoding):
      labels = encoding['labels']
      labels[labels == -100] = processor.tokenizer.pad_token_id
      label_str = processor.decode(labels, skip_special_tokens=True)
      return label_str

    get_label_str(train_dataset[0])


    # 2.3 Setup Loss Metrics
    from evaluate import load
    cer_metric = load("cer")
    
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
    
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
        return {"cer": cer}


    # 2.4 Model Configuration
    # Analyze your dataset first
    avg_target_len = df_train['text'].apply(len).mean()
    print("average target length", avg_target_len)
    max_target_len = int(df_train['text'].apply(len).quantile(0.95))
    print("maximum target length", max_target_len)


    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        num_train_epochs=1,
        max_steps=1000, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        fp16=True,
        output_dir=args.model_dir,
        logging_steps=500,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False
    )

    from transformers import default_data_collator
    
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()


