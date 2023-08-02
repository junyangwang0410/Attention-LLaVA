from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
import json
from PIL import Image
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--image', type=str)
parser.add_argument('--prompt', type=str, default="Describe this image.")
parser.add_argument('--output', type=str)
parser.add_argument('--max-length', type=int, default=64)
parser.add_argument('--layer', type=int, default=32)
args = parser.parse_args()


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model():
    disable_torch_init()
    model_name = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    qs = args.prompt
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN # system(1-97) <im_start>(98) <im_token>*256(354) <im_end>(355) text_start(360)
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    conv_mode = "multimodal"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    image_file = args.image
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                max_new_tokens=args.max_length,
                stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    an = outputs.strip()

    conv_mode = "multimodal"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], an)
    prompt = conv.get_prompt()[:-3]
    inputs = tokenizer([prompt])
    
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.no_grad():
        output = model(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            output_attentions=True,
            return_dict=True,
            )

    attention = torch.mean(output.attentions[args.layer-1].squeeze(0), dim=0)
    attention = attention[358:, :]
    attention_image = attention[:, 98:354]
    logits = output.logits[:, 358:, :]
    for i in range(len(attention)):
        logit = logits[:, i, :]
        token_id = torch.argmax(logit, dim=1).item()
        token = tokenizer.decode([token_id], skip_special_tokens=True)

        attention_hallu = attention[i, 98:354]
        attention_hallu = torch.softmax(attention_hallu*200, dim=0).view(16, 16)
        attention_hallu = np.array(attention_hallu.cpu(), dtype=np.float32)*100

        img = Image.open(image_file)
        resized_attention = np.array(Image.fromarray((attention_hallu*255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR))
        smoothed_attention = gaussian_filter(resized_attention, sigma=2)
        heatmap = np.uint8(smoothed_attention)
        heatmap = plt.cm.jet(heatmap)[:, :, :3] * 255
        heatmap = Image.fromarray(np.uint8(heatmap)).resize(img.size)
        result = Image.blend(img.convert('RGBA'), heatmap.convert('RGBA'), alpha=0.5)
        result.convert('RGB').save(args.output + f'/{i+1}_{token}.jpg')

    attention_image = torch.mean(attention_image, dim=1).unsqueeze(1)
    attention = torch.cat([attention_image, attention[:, 359:]], dim=1).T
    l = len(attention)
    y = ["<Image>"]
    x = []
    for i in range(l-1):
        logits = output.logits[:, 358+i, :]
        token_id = torch.argmax(logits, dim=1).item()
        token = tokenizer.decode([token_id], skip_special_tokens=True)
        y.append(token)
        x.append(token)
    y = y[:-1]
    attention = attention.tolist()
    for i in range(len(attention[0])):
        attention[0][i] *= 1
    sns.heatmap(attention, cmap="Blues", xticklabels=x, yticklabels=y)
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(args.output + "/heatmap.png", dpi=200)


if __name__ == "__main__":
    eval_model()
