# Towards Merging Semantic Segmentation and Language-guided Saliency Prediction

The repo is the implementation for Project 2 in course CS3324, SJTU, lectured by Xiongkuo Min and Guangtao Zhai.

Most of the codes are originated from this repo [lang-seg](https://github.com/isl-org/lang-seg)

The checkpoints and the detailed training code will be filled in soon!

## Usage
### Installation
Option 1: 

``` pip install -r requirements.txt ```

Option 2: 
```
conda install ipython
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==1.3.5
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install altair
pip install streamlit
pip install --upgrade protobuf
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
```

### Data Preparation


### Training and Testing Example
Training: Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

