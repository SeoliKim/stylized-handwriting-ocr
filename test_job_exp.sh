#!/bin/bash
#SBATCH --job-name=test_example              # Job name
#SBATCH --ntasks=1                         # Number of tasks (1 process node)
#SBATCH --gres=gpu:1                       # Request 1 GPU for testing (adjust if needed)
#SBATCH --mem=256G                          
#SBATCH --time=12:00:00                     
#SBATCH --output=test_base_output.log           
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@example.com

# Activate your Python environment
source ./venv/bin/activate

# Run your testing script (single GPU typically sufficient for testing)
python test_script.py \
    --model-name "microsoft/trocr-base-handwritten" \
    --processor-name "microsoft/trocr-base-handwritten" \
    --result-path "eval/result_base.csv" \
    --images-path "/common/users/$user/df_words" \
    --project-path "/common/home/stylized-ocr"
