from typing import List

from PIL import Image
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.utils import unwrap_model

from minigpt4.datasets.builders import *
from minigpt4.models import MiniGPT4
from minigpt4.processors import Blip2ImageEvalProcessor

from quantize_linear import load_quant


SYSTEM_PROMPT = "Give the following image: <Img>ImageContent</Img>. " + \
    "You will be able to see the image once I provide it to you. Please answer my question in short.\n"


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class EvalModel(BaseEvalModel):

    def __init__(self, model_args):
        assert (
            "lm_path" in model_args and "minigpt4_path" in model_args
        ), "MiniGPT-4 requires lm_path, minigpt4_path, and device arguments to be specified"

        self.model = MiniGPT4(llama_model=model_args["lm_path"])
        print("Load BLIP2-LLM Checkpoint: {}".format(model_args["minigpt4_path"]))
        ckpt = torch.load(model_args["minigpt4_path"], map_location="cpu")
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model.llama_tokenizer.padding_side = "right"

        self.vis_processor = Blip2ImageEvalProcessor(image_size=224)
    
        stop_words_ids = [torch.tensor([869]).cuda()]  # '.'
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        if "quant_checkpoint" in model_args:
            if "smooth_checkpoint" not in model_args:
                model_args["smooth_checkpoint"] = None
            self.model.llama_model = load_quant(
                self.model.llama_model,
                model_args["quant_checkpoint"],
                model_args["quant_wbits"],
                model_args["quant_abits"],
                model_args["smooth_checkpoint"],
            )

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
        max_length: int = 2000,
    ) -> List[str]:
        model = unwrap_model(self.model)
        assert all(len(example) == 1 for example in batch_images), "BLIP-2 only supports one image per example"
        batch_images = torch.stack([
            self.vis_processor(example[0].convert("RGB"))
            for example in batch_images
        ]).to(model.device)
        img_embeds, atts_img = model.encode_img(batch_images)
        # vqa_prompt = f"{SYSTEM_PROMPT}###Human: <Img><ImageHere></Img>\n"
        vqa_prompt = f"Image: <ImageHere> "
        img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, vqa_prompt)

        to_regress_tokens = model.llama_tokenizer(
            batch_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        ).to(model.device)

        batch_size = img_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * model.llama_tokenizer.bos_token_id
        bos_embeds = model.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        for i, shift in enumerate(attention_mask.shape[-1] - attention_mask.sum(-1)):
            inputs_embeds[i] = inputs_embeds[i].roll((shift.item()), dims=(0))
        attention_mask = attention_mask.flip(-1)

        with torch.inference_mode():
            outputs = model.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_generation_length,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_generation_length,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=length_penalty,
                temperature=1.0,
            )

        return model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
        # return f"###Human: {question}\n###Assistant: "
        return f"Question: {question} Short answer: {answer if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"A photo of {caption if caption is not None else ''}"

    def get_imagenet_prompt(self, label=None) -> str:
        raise NotImplementedError

    def get_hateful_memes_prompt(self, text, label=None) -> str:
        raise NotImplementedError
