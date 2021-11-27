# TVT
Code of [TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2108.05988.pdf)

<p align="left"> 
<img width="400" src="https://github.com/uta-smile/TVT/blob/main/image.png">
</p>

### Pretrained ViT

- Download [ViT-B_16.npz](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007) and put it at `checkpoint/ViT-B_16.npz`

### Datasets:

- Download [data](https://drive.google.com/file/d/1loJn7B0wBLdtkvhHtLpBnkKrJm7REnJI/view?usp=sharing) and replace the current `data/`

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`

### Training:

An example:
```
python3 main.py --train_batch_size 64 --dataset office --name wa \
--source_list data/office/webcam_list.txt --target_list data/office/amazon_list.txt \
--test_list data/office/amazon_list.txt --num_classes 31 --model_type ViT-B_16 \
--pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 \
--beta 0.1 --gamma 0.01 --use_im --theta 0.1
```


Citation:
```
@article{yang2021tvt,
  title={TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation},
  author={Yang, Jinyu and Liu, Jingjing and Xu, Ning and Huang, Junzhou},
  journal={arXiv preprint arXiv:2108.05988},
  year={2021}
}
```
Our code is largely borrowed from [CDAN](https://github.com/thuml/CDAN) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
