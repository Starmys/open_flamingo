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
    --model llava \
    --model_base none \
    --model_path /home/chengzhang/models/llava/llava-v1.5-7b \
    --results_file "/home/chengzhang/Multimodal-Quantization/evaluation/LLaVA/results-1.5-w16a16-textvqa.json" \
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
    --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"

# python open_flamingo/eval/evaluate.py \
#     --model llava \
#     --model_base /home/chengzhang/models/llava/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
#     --model_path /home/chengzhang/models/llava/llava-336px-pretrain-vicuna-13b-v1.3 \
#     --results_file "/home/chengzhang/Multimodal-Quantization/evaluation/LLaVA/results-proj-w4a8-textvqa.json" \
#     --precision fp16 \
#     --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-viquna-13b-textvqa-4bit.pt \
#     --quant_wbits 4 \
#     --quant_abits 8 \
#     --batch_size 4 \
#     --shots 0 \
#     --eval_ok_vqa \
#     --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#     --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#     --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#     --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#     --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#     --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#     --eval_textvqa \
#     --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"

# python open_flamingo/eval/evaluate.py \
#     --model llava \
#     --model_base /home/chengzhang/models/llava/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
#     --model_path /home/chengzhang/models/llava/llava-336px-pretrain-vicuna-13b-v1.3 \
#     --results_file "/home/chengzhang/Multimodal-Quantization/evaluation/LLaVA/results-proj-w4a16-textvqa.json" \
#     --precision fp16 \
#     --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-viquna-13b-textvqa-4bit.pt \
#     --quant_wbits 4 \
#     --quant_abits 32 \
#     --batch_size 4 \
#     --shots 0 \
#     --eval_ok_vqa \
#     --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#     --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#     --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#     --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#     --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#     --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#     --eval_textvqa \
#     --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"

# python open_flamingo/eval/evaluate.py \
#     --model llava \
#     --model_base /home/chengzhang/models/llava/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
#     --model_path /home/chengzhang/models/llava/llava-336px-pretrain-vicuna-13b-v1.3 \
#     --results_file "/home/chengzhang/Multimodal-Quantization/evaluation/LLaVA/results-proj-w8a4.json" \
#     --precision fp16 \
#     --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-viquna-13b-8bit.pt \
#     --quant_wbits 8 \
#     --quant_abits 4 \
#     --batch_size 4 \
#     --shots 0 \
#     --eval_ok_vqa \
#     --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#     --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#     --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#     --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#     --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#     --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#     --eval_textvqa \
#     --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"

# python open_flamingo/eval/evaluate.py \
#     --model llava \
#     --model_base /home/chengzhang/models/llava/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
#     --model_path /home/chengzhang/models/llava/llava-336px-pretrain-vicuna-13b-v1.3 \
#     --results_file "/home/chengzhang/Multimodal-Quantization/evaluation/LLaVA/results-proj-w4a4.json" \
#     --precision fp16 \
#     --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-viquna-13b-4bit.pt \
#     --quant_wbits 4 \
#     --quant_abits 4 \
#     --batch_size 4 \
#     --shots 0 \
#     --eval_ok_vqa \
#     --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#     --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#     --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#     --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#     --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#     --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#     --eval_textvqa \
#     --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
#     # --eval_vqav2 \
#     # --vqav2_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/image/train2014" \
#     # --vqav2_train_annotations_json_path "/home/chengzhang/datasets/VQAv2/annotation/v2_mscoco_train2014_annotations.json" \
#     # --vqav2_train_questions_json_path "/home/chengzhang/datasets/VQAv2/question/v2_OpenEnded_mscoco_train2014_questions.json" \
#     # --vqav2_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/image/val2014" \
#     # --vqav2_test_annotations_json_path "/home/chengzhang/datasets/VQAv2/annotation/v2_mscoco_val2014_annotations.json" \
#     # --vqav2_test_questions_json_path "/home/chengzhang/datasets/VQAv2/question/v2_OpenEnded_mscoco_val2014_questions.json" \
