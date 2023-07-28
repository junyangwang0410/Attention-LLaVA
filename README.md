# Attention-LLaVA
A hot-pluggable tool for visualizing LLaVA's attention.

## Example

### Dialog
<p align="center">
  <a><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/case.png" width="70%"></a> <br>
</p>

### Heatmap
<p align="center">
  <a><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/heatmap.png" width="70%"></a> <br>
</p>

### Attention
<p align="center">
  <a><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/1_The.jpg" width="20%"><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/2_image.jpg" width="20%"><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/3_features.jpg" width="20%"><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/4_a.jpg" width="20%"><img src="https://github.com/junyangwang0410/Attention-LLaVA/blob/main/5_woman.jpg" width="20%"></a> <br>
The attention on the image of the first 5 words which are "The image features a woman".
</p>

## Usage
1. Install LLaVA from [Link](https://github.com/haotian-liu/LLaVA).
2. Put the [attention.py](https://github.com/junyangwang0410/Attention-LLaVA/blob/main/attention.py) into:
```
LLaVA-main/llava/eval
```
3. Run by this command:
```
cd LLaVA-main
python -m llava.eval.attention \
  --checkpoint path/to/llava/checkpoint \
  --image path/to/image \
  --layer 32 \
  --output path/to/output/result \
  --max-length 64
```
