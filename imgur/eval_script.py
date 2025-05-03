#!/usr/bin/env python
# coding: utf-8

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TROCR Model Evaluation")
    parser.add_argument(
        "--project-dir",
        type=str,
        default='/common/users/$id/project',
        help="Path to the root of the project"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default='dataset',
        help="Relative Path to the dataset"
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
        "--result-path",
        type=str,
        default="eval/result.csv",
        help="Path to save evaluation results"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Now use args.model_name, args.processor_name, args.result_path in your code
    print(f"Using model: {args.model_name}")
    print(f"Using processor: {args.processor_name}")
    print(f"Results will be saved to: {args.result_path}")
    
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
    
    
    # Merge into one dataframe
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
    
    
    # Confirm images are correctly loaded
    loaded_dfwords.info()
    loaded_dfwords.head(3)
    
    
    # 1.2 Loading the Testing Data
    df_test_info= pd.read_csv(args.dataset_dir+"/dataset_info/df_test_info.csv")
    df_test_info.head()

    # extract the testing portion from image dataset
    df_test_info_premerge= df_test_info.drop(['id', 'text'], axis=1)
    df_test= pd.merge(
        loaded_dfwords.reset_index(), 
        df_test_info_premerge, 
        left_on='index', 
        right_on='word_idx',
        how='right' 
    )
    
    # Check if testing dataset is loaded correctly
    df_test.info()
    df_test.head()

    # Step 2. Make Inference of the IAM-fine-tuned Base Model
    
    # 2.1 Setup Target Model
    # get targetted model
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(args.processor_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    
    
    # 2.2 Do Inference- OCRing
    from tqdm import tqdm
    from PIL import Image
    def readText_batch(df, indices, model, processor):
        """Process multiple images at once"""
        images= [df['image'][idx]for idx in indices]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    def process_all_rows_batched(df, model, processor, batch_size=8):
        results = []
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_indices = range(i, min(i+batch_size, len(df)))
            try:
                batch_texts = readText_batch(df, batch_indices,model, processor)
                for idx, text in zip(batch_indices, batch_texts):
                    results.append({
                        'id': df['id'][idx],
                        'true_text': df['text'][idx],
                        'predicted_text': text
                    })
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {str(e)}")
                for idx in batch_indices:
                    results.append({
                        'id': df['id'][idx],
                        'true_text': df['text'][idx],
                        'predicted_text': None,
                        'error': str(e)
                    })
        return pd.DataFrame(results)
    
    # Run
    results_df = process_all_rows_batched(df_test, model, processor)
    
    
    # Step 3. Evalute its performance
    
    # 3.1 Define Metrics
    from tqdm import tqdm
    from evaluate import load
    cer = load("cer")
    
    def compute_metrics(pred_str, label_str):
        pred_str=pred_str.strip()
        label_str=label_str.strip()
        try: 
            score = cer.compute(predictions=[pred_str], references=[label_str])
            return score
        except Exception as e:
            print("error", e)
            print(type(pred_str), len(pred_str), pred_str)
            print(type(label_str), len(label_str), label_str)
            return None

    from tqdm import tqdm
    tqdm.pandas()  
    # Run evalution
    results_df["metrics"] = results_df.progress_apply(
        lambda row: compute_metrics(row["predicted_text"], row["true_text"]),
        axis=1
    )
    
    # Save evlaution result
    results_df.to_csv(args.result_path, index=False)

if __name__ == "__main__":
    main()
