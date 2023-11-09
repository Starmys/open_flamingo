from typing import List

from PIL import Image
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.utils import unwrap_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from quantize_linear import load_quant


class EvalModel(BaseEvalModel):

    def __init__(self, model_args):
        assert (
            "model_path" in model_args and "model_base" in model_args
        ), "LLaVA requires model_path, model_base, and device arguments to be specified"

        if model_args["model_base"] == 'none':
            model_args["model_base"] = None

        model_name = get_model_name_from_path(model_args["model_path"])
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_args["model_path"], model_args["model_base"], model_name
        )
        self.model.model.requires_grad_(True)

        # import ipdb; ipdb.set_trace()
        if "quant_args" in model_args:
            quant_args = {k: v for k, v in [x.split('=') for x in model_args['quant_args'].replace('"', '').split(',')]}
            print(f'[Quant Args] {quant_args}')
            self.model = load_quant(self.model, **quant_args)

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        model = unwrap_model(self.model)
        assert all(len(example) == 1 for example in batch_images), "LLaVA only supports one image per example"

        batch_images = torch.stack([
            self.image_processor.preprocess(example[0].convert("RGB"), return_tensors='pt')['pixel_values'].half()
            for example in batch_images
        ]).to(model.device)

        input_ids = [
            tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX)
            for text in batch_text
        ]
        # print(input_ids)
        max_len = max(len(seq) for seq in input_ids)
        input_ids = [
            [self.tokenizer.pad_token_id] * (max_len - len(seq)) + seq
            for seq in input_ids
        ]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=batch_images,
                do_sample=True,
                # temperature=0.0,
                min_length=min_generation_length,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                use_cache=True,
                # stopping_criteria=self.stopping_criteria,
            )

        # import ipdb; ipdb.set_trace()
        batch_output = self.tokenizer.batch_decode(output_ids[:, max_len:], skip_special_tokens=True)
        return [seq.split('\n')[0].split('.')[0].split(',')[0] for seq in batch_output]

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        raise NotImplementedError

    def get_vqa_prompt(self, question, answer=None) -> str:
        # sys = "A chat between a curious user and an artificial intelligence assistant. "
        # sys += "The assistant shortly answers to the user's question in English."
        # return f"{sys} USER: {question} ASSISTANT: "
        return f"Image: <image> Question: {question} Brief answer: {answer if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"

    def get_imagenet_prompt(self, label=None) -> str:
        raise NotImplementedError

    def get_hateful_memes_prompt(self, text, label=None) -> str:
        raise NotImplementedError
