import json
import random

import torch
import numpy as np
from PIL import Image

from open_flamingo import create_model_and_transforms


PREFIX = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def setup_seeds(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    setup_seeds()

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="../Multimodal-GPT/checkpoints/llama-7b_hf",
        tokenizer_path="../Multimodal-GPT/checkpoints/llama-7b_hf",
        cross_attn_every_n_layers=4
    )
    model.load_state_dict(torch.load("../Multimodal-GPT/checkpoints/OpenFlamingo-9B/checkpoint.pt"), strict=False)

    with open('../datasets/OK-VQA/question/OpenEnded_mscoco_val2014_questions.json') as f:
        questions = json.loads(f.read())['questions']

    with open('results.txt', 'w') as f:
        f.write('')

    tokenizer.padding_side = "left" # For generation padding tokens should be on the left

    for q in questions:
        # try:
        img_id = q['image_id']
        img_path = f'../datasets/OK-VQA/image/val2014/COCO_val2014_{str(img_id).zfill(12)}.jpg'
        img = Image.open(img_path)
        vision_x = torch.cat([image_processor(img).unsqueeze(0)], dim=0).unsqueeze(1).unsqueeze(0)
        # vision_x = image_processor(img).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        question = q['question']
        lang_x = tokenizer(
            [f"{PREFIX}\n### Image: <image>\n### Instruction: {question}\n### Response:"],
            # [f"<image>\n{question}\n"],
            return_tensors="pt",
        )
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=1,
        )
        result = tokenizer.decode(generated_text[0])#.split('### Response:')[-1]
        # except RuntimeError:
        #     continue
        with open('results.txt', 'a') as f:
            f.write('#' + str(q['question_id']) + '\n' + result + '\n\n')
