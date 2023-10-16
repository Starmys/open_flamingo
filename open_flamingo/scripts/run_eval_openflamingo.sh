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
# python open_flamingo/eval/evaluate.py \
#     --model open_flamingo \
#     --vision_encoder_path ViT-L-14 \
#     --vision_encoder_pretrained openai \
#     --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#     --lm_tokenizer_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#     --cross_attn_every_n_layers 4 \
#     --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-deprecated/checkpoint.pt \
#     --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-llama-tmp.json \
#     --precision fp16 \
#     --batch_size 8 \
#     --shots 0 \
#     --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-llama-textvqa-4bit.pt \
#     --quant_wbits 4 \
#     --quant_abits 32 \
#     --ignore_components decoder_layer.self_attn.q_proj,decoder_layer.self_attn.k_proj,decoder_layer.self_attn.v_proj \
#     --eval_ok_vqa \
#     --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#     --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#     --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#     --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#     --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#     --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
    # --eval_textvqa \
    # --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images" \
    # --eval_vqav2 \
    # --vqav2_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/image/train2014" \
    # --vqav2_train_annotations_json_path "/home/chengzhang/datasets/VQAv2/annotation/v2_mscoco_train2014_annotations.json" \
    # --vqav2_train_questions_json_path "/home/chengzhang/datasets/VQAv2/question/v2_OpenEnded_mscoco_train2014_questions.json" \
    # --vqav2_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/image/val2014" \
    # --vqav2_test_annotations_json_path "/home/chengzhang/datasets/VQAv2/annotation/v2_mscoco_val2014_annotations.json" \
    # --vqav2_test_questions_json_path "/home/chengzhang/datasets/VQAv2/question/v2_OpenEnded_mscoco_val2014_questions.json" \

count=0
for comp in "decoder_layer.ffn.up_proj" "decoder_layer.ffn.down_proj" "decoder_layer.ffn.up_proj decoder_layer.ffn.down_proj"; do
  for abits in "8" "16"; do
    python open_flamingo/eval/evaluate.py \
        --model open_flamingo \
        --vision_encoder_path ViT-L-14 \
        --vision_encoder_pretrained openai \
        --lm_path /home/chengzhang/models/mpt/mpt-7b \
        --lm_tokenizer_path /home/chengzhang/models/mpt/mpt-7b \
        --cross_attn_every_n_layers 4 \
        --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
        --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w4a${abits}-mpt-ig0-${count}.json \
        --precision fp16 \
        --batch_size 8 \
        --shots 0 \
        --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-mpt-textvqa-4bit.pt \
        --w_bits 4 \
        --a_bits "${abits}" \
        --ignore_layers 0 \
        --ignore_components "${comp}" \
        --eval_ok_vqa \
        --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
        --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
        --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
        --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
        --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
        --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
        --eval_textvqa \
        --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
    (( count++ ))
  done
done

# for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
#   python open_flamingo/eval/evaluate.py \
#       --model open_flamingo \
#       --vision_encoder_path ViT-L-14 \
#       --vision_encoder_pretrained openai \
#       --lm_path /home/chengzhang/models/mpt/mpt-7b \
#       --lm_tokenizer_path /home/chengzhang/models/mpt/mpt-7b \
#       --cross_attn_every_n_layers 4 \
#       --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
#       --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/ignore-layer/results-w4a16-mpt-ig${layer}.json \
#       --precision fp16 \
#       --batch_size 8 \
#       --shots 0 \
#       --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-mpt-textvqa-4bit.pt \
#       --quant_wbits 4 \
#       --quant_abits 32 \
#       --ignore_layers ${layer} \
#       --eval_ok_vqa \
#       --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#       --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#       --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#       --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#       --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#       --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#       --eval_textvqa \
#       --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
# done

# for comp in "gated_cross_attn_layer.attn.to_q" "gated_cross_attn_layer.attn.to_kv" "gated_cross_attn_layer.attn.to_out" "gated_cross_attn_layer.ff.1" "gated_cross_attn_layer.ff.3" "decoder_layer.attn.Wqkv" "decoder_layer.attn.out_proj" "decoder_layer.ffn.up_proj" "decoder_layer.ffn.down_proj"; do
#   python open_flamingo/eval/evaluate.py \
#       --model open_flamingo \
#       --vision_encoder_path ViT-L-14 \
#       --vision_encoder_pretrained openai \
#       --lm_path /home/chengzhang/models/mpt/mpt-7b \
#       --lm_tokenizer_path /home/chengzhang/models/mpt/mpt-7b \
#       --cross_attn_every_n_layers 4 \
#       --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
#       --results_file //home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/ignore-component/results-w4a16-mpt-${comp}.json \
#       --precision fp16 \
#       --batch_size 8 \
#       --shots 0 \
#       --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-mpt-textvqa-4bit.pt \
#       --quant_wbits 4 \
#       --quant_abits 32 \
#       --ignore_components ${comp} \
#       --eval_ok_vqa \
#       --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#       --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#       --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#       --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#       --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#       --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#       --eval_textvqa \
#       --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
# done

# gated_cross_attn_layer.attn.['to_q', 'to_kv', 'to_out']
# gated_cross_attn_layer.ff.[1, 3]
# decoder_layer.attn.['Wqkv', 'out_proj']
# decoder_layer.ffn.['up_proj', 'down_proj']

# for wbits in 8 4; do
#   for abits in 32 8 4; do
#     echo "==================== w${wbits}a${abits}-mpt-of ===================="
#     python open_flamingo/eval/evaluate.py \
#         --model open_flamingo \
#         --vision_encoder_path ViT-L-14 \
#         --vision_encoder_pretrained openai \
#         --lm_path /home/chengzhang/models/mpt/mpt-7b \
#         --lm_tokenizer_path /home/chengzhang/models/mpt/mpt-7b \
#         --cross_attn_every_n_layers 4 \
#         --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
#         --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w${wbits}a${abits}-mpt.json \
#         --precision fp16 \
#         --batch_size 8 \
#         --shots 0 \
#         --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-mpt-textvqa-${wbits}bit.pt \
#         --quant_wbits ${wbits} \
#         --quant_abits ${abits} \
#         --eval_ok_vqa \
#         --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#         --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#         --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#         --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#         --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#         --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#         --eval_textvqa \
#         --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
#   done
# done

# for wbits in 8 4; do
#   for abits in 8 4; do
#     echo "==================== w${wbits}a${abits}-sf-mpt-of ===================="
#     python open_flamingo/eval/evaluate.py \
#         --model open_flamingo \
#         --vision_encoder_path ViT-L-14 \
#         --vision_encoder_pretrained openai \
#         --lm_path /home/chengzhang/models/mpt/mpt-7b \
#         --lm_tokenizer_path /home/chengzhang/models/mpt/mpt-7b \
#         --cross_attn_every_n_layers 4 \
#         --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
#         --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w${wbits}a${abits}-sf-mpt-of.json \
#         --precision fp16 \
#         --batch_size 8 \
#         --shots 0 \
#         --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-mpt-textvqa-${wbits}bit.pt \
#         --quant_wbits ${wbits} \
#         --quant_abits ${abits} \
#         --smooth_checkpoint fake \
#         --eval_ok_vqa \
#         --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#         --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#         --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#         --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#         --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#         --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#         --eval_textvqa \
#         --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
#   done
# done

# echo "==================== w16a16-sf-mpt-of ===================="
# python open_flamingo/eval/evaluate.py \
#     --model open_flamingo \
#     --vision_encoder_path ViT-L-14 \
#     --vision_encoder_pretrained openai \
#     --lm_path /home/chengzhang/models/mpt/mpt-7b \
#     --lm_tokenizer_path /home/chengzhang/models/mpt/mpt-7b \
#     --cross_attn_every_n_layers 4 \
#     --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
#     --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w16a16-mpt-of.json \
#     --precision fp16 \
#     --batch_size 8 \
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






# echo "############################################################"






# for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
#   python open_flamingo/eval/evaluate.py \
#       --model open_flamingo \
#       --vision_encoder_path ViT-L-14 \
#       --vision_encoder_pretrained openai \
#       --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#       --lm_tokenizer_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#       --cross_attn_every_n_layers 4 \
#       --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-deprecated/checkpoint.pt \
#       --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/ignore-layer/results-w4a16-llama-ig${layer}.json \
#       --precision fp16 \
#       --batch_size 8 \
#       --shots 0 \
#       --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-llama-textvqa-4bit.pt \
#       --quant_wbits 4 \
#       --quant_abits 32 \
#       --ignore_layers ${layer} \
#       --eval_ok_vqa \
#       --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#       --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#       --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#       --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#       --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#       --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#       --eval_textvqa \
#       --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
# done

# for comp in "gated_cross_attn_layer.attn.to_q" "gated_cross_attn_layer.attn.to_kv" "gated_cross_attn_layer.attn.to_out" "gated_cross_attn_layer.ff.1" "gated_cross_attn_layer.ff.3" "decoder_layer.self_attn.q_proj,decoder_layer.self_attn.k_proj,decoder_layer.self_attn.v_proj" "decoder_layer.self_attn.o_proj" "decoder_layer.mlp.gate_proj,decoder_layer.mlp.up_proj" "decoder_layer.mlp.down_proj"; do
#   python open_flamingo/eval/evaluate.py \
#       --model open_flamingo \
#       --vision_encoder_path ViT-L-14 \
#       --vision_encoder_pretrained openai \
#       --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#       --lm_tokenizer_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#       --cross_attn_every_n_layers 4 \
#       --checkpoint_path /home/chengzhang/models/openflamingo/OpenFlamingo-9B-deprecated/checkpoint.pt \
#       --results_file //home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/ignore-component/results-w4a16-llama-${comp}.json \
#       --precision fp16 \
#       --batch_size 8 \
#       --shots 0 \
#       --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-llama-textvqa-4bit.pt \
#       --quant_wbits 4 \
#       --quant_abits 32 \
#       --ignore_components ${comp} \
#       --eval_ok_vqa \
#       --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#       --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#       --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#       --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#       --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#       --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#       --eval_textvqa \
#       --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
# done

# for wbits in 8 4; do
#   for abits in 32 8; do
#     echo "==================== w${wbits}a${abits}-llama-of ===================="
#     python open_flamingo/eval/evaluate.py \
#         --model open_flamingo \
#         --vision_encoder_path ViT-L-14 \
#         --vision_encoder_pretrained openai \
#       --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#       --lm_tokenizer_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#       --cross_attn_every_n_layers 4 \
#       --checkpoint_path "/home/chengzhang/models/openflamingo/OpenFlamingo-9B-deprecated/checkpoint.pt" \
#         --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w${wbits}a${abits}-llama.json \
#         --precision fp16 \
#         --batch_size 8 \
#         --shots 0 \
#         --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-llama-textvqa-${wbits}bit.pt \
#         --quant_wbits ${wbits} \
#         --quant_abits ${abits} \
#         --eval_ok_vqa \
#         --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#         --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#         --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#         --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#         --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#         --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#         --eval_textvqa \
#         --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
#   done
# done

# for wbits in 8 4; do
#   for abits in 8 4; do
#     echo "==================== w${wbits}a${abits}-sf-llama-of ===================="
#     python open_flamingo/eval/evaluate.py \
#         --model open_flamingo \
#         --vision_encoder_path ViT-L-14 \
#         --vision_encoder_pretrained openai \
#         --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#         --lm_tokenizer_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#         --cross_attn_every_n_layers 4 \
#         --checkpoint_path "/home/chengzhang/models/openflamingo/OpenFlamingo-9B-deprecated/checkpoint.pt" \
#         --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w${wbits}a${abits}-sf-llama-of.json \
#         --precision fp16 \
#         --batch_size 8 \
#         --shots 0 \
#         --quant_checkpoint /home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/openflamingo-9b-llama-textvqa-${wbits}bit.pt \
#         --quant_wbits ${wbits} \
#         --quant_abits ${abits} \
#         --smooth_checkpoint fake \
#         --eval_ok_vqa \
#         --ok_vqa_train_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/train2014" \
#         --ok_vqa_train_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_train2014_annotations.json" \
#         --ok_vqa_train_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_train2014_questions.json" \
#         --ok_vqa_test_image_dir_path "/home/chengzhang/datasets/OK-VQA/images/val2014" \
#         --ok_vqa_test_annotations_json_path "/home/chengzhang/datasets/OK-VQA/annotation/mscoco_val2014_annotations.json" \
#         --ok_vqa_test_questions_json_path "/home/chengzhang/datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json" \
#         --eval_textvqa \
#         --textvqa_image_dir_path "/home/chengzhang/datasets/TextVQA/images/train_images"
#   done
# done

# echo "==================== w16a16-sf-llama-of ===================="
# python open_flamingo/eval/evaluate.py \
#     --model open_flamingo \
#     --vision_encoder_path ViT-L-14 \
#     --vision_encoder_pretrained openai \
#     --lm_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#     --lm_tokenizer_path /home/chengzhang/models/llama-hf/llama-7b_hf \
#     --cross_attn_every_n_layers 4 \
#     --checkpoint_path "/home/chengzhang/models/openflamingo/OpenFlamingo-9B-deprecated/checkpoint.pt" \
#     --results_file /home/chengzhang/Multimodal-Quantization/evaluation/Open-Flamingo/results-w16a16-llama-of.json \
#     --precision fp16 \
#     --batch_size 8 \
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
