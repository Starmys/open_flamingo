#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

<<com
Example Slurm evaluation script. 
Notes:
- VQAv2 test-dev and test-std annotations are not publicly available. 
  To evaluate on these splits, please follow the VQAv2 instructions and submit to EvalAI.
  This script will evaluate on the val split.
com

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8000
export WORLD_SIZE=1
export COUNT_NODE=1

export PYTHONPATH="$PYTHONPATH:open_flamingo"
python open_flamingo/eval/evaluate.py \
    --model minigpt4 \
    --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
    --minigpt4_path /home/chengzhang/models/minigpt4/prerained_minigpt4_7b.pth\
    --results_file "/home/chengzhang/Multimodal-Quantization/evaluation/MiniGPT-4/results-w16a16.json" \
    --precision fp16 \
    --batch_size 8 \
    --shots 0 \
    --eval_ok_vqa \
    --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
    --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
    --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
    --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
    --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
    --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
    --eval_textvqa \
    --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images" \
    # --eval_vqav2 \
    # --vqav2_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/image/train2014" \
    # --vqav2_train_annotations_json_path "/home/chengzhang/datasets/VQAv2/annotation/v2_mscoco_train2014_annotations.json" \
    # --vqav2_train_questions_json_path "/home/chengzhang/datasets/VQAv2/question/v2_OpenEnded_mscoco_train2014_questions.json" \
    # --vqav2_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/image/val2014" \
    # --vqav2_test_annotations_json_path "/home/chengzhang/datasets/VQAv2/annotation/v2_mscoco_val2014_annotations.json" \
    # --vqav2_test_questions_json_path "/home/chengzhang/datasets/VQAv2/question/v2_OpenEnded_mscoco_val2014_questions.json" \
