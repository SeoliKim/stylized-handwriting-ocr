#!/bin/bash
#SBATCH --job-name=train_example              # Job name
#SBATCH --ntasks=1                         # Number of tasks (1 process node)
#SBATCH --gres=gpu:1                       # Request 1 GPU for testing (adjust if needed)
#SBATCH --mem=500G                          # Memory (reduced for testing)
#SBATCH --time=20:00:00                     # Shorter time limit for testing
#SBATCH --output=train_output.log           # Test log file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@example.com

# Activate your Python environment
source ./venv/bin/activate

# Run your testing script (single GPU typically sufficient for testing)
python train_script.py \
    --model-name "microsoft/trocr-small-stage1" \
    --processor-name "microsoft/trocr-small-stage1" \
    --output-dir "models/my" \
    --images-dir "/common/users/$user/df_words" \
    --project-dir "/common/home/stylized-ocr"
