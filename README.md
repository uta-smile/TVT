# [TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.05988)

### updates (03/10/2022)
1. Add the environment requirements to reproduce the results.

2. Add the attention visualization code. An example is as follows where `att_visual.txt` contains image pathes:
```
python3 visualize.py --dataset office --name dw --num_classes 31 --image_path att_visual.txt --img_size 256
```
More details can be found in [Attention Map Visualization](https://github.com/uta-smile/TVT/blob/main/README.md#attention-map-visualization)

### updates (03/08/2022)
Add the source-only code. An example on `Office-31` dataset is as follows, where `dslr` is the source domain, `webcam` is the target domain:
```
python3 train.py --train_batch_size 64 --dataset office --name dw_source_only --train_list data/office/dslr_list.txt --test_list data/office/webcam_list.txt --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256
```

<p align="left"> 
<img width="400" src="https://github.com/uta-smile/TVT/blob/main/image.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install tqdm==4.50.2
pip install tensorboard==2.8.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.5.2
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
```

### Pretrained ViT
Download the following models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)
- ViT-B_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-B_16.npz;tab=live_object)

TVT with ViT-B_16 (ImageNet-21K) performs a little bit better than TVT with ViT-B_16 (ImageNet):
<p align="left"> 
<img width="500" src="https://github.com/uta-smile/TVT/blob/main/ImageNet_vs_ImageNet21K.png">
</p>

### Datasets:

- Download [data](https://drive.google.com/file/d/1rnU49vEEdtc3EYVo7QydWzxcSuYqZbUB/view?usp=sharing) and replace the current `data/`

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`

### Training:

All commands can be found in `script.txt`. An example:
```
python3 main.py --train_batch_size 64 --dataset office --name wa \
--source_list data/office/webcam_list.txt --target_list data/office/amazon_list.txt \
--test_list data/office/amazon_list.txt --num_classes 31 --model_type ViT-B_16 \
--pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 \
--beta 0.1 --gamma 0.01 --use_im --theta 0.1
```

### Attention Map Visualization:
```
python3 visualize.py --dataset office --name wa --num_classes 31 --image_path att_visual.txt --img_size 256
```
The code will automatically use the best model in `wa` to visualize the attention maps of images in `att_visual.txt`. `att_visual.txt` contains image pathes you want to visualize, for example:
```
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0001.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0002.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0003.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0004.jpg 5
/data/office/domain_adaptation_images/dslr/images/calculator/frame_0005.jpg 5
```


### Citation:
```
@article{yang2021tvt,
  title={TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation},
  author={Yang, Jinyu and Liu, Jingjing and Xu, Ning and Huang, Junzhou},
  journal={arXiv preprint arXiv:2108.05988},
  year={2021}
}
```
Our code is largely borrowed from [CDAN](https://github.com/thuml/CDAN) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
