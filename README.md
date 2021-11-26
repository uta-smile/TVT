# TVT
Code of [TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2108.05988.pdf)

<p align="left"> 
<img width="400" src="https://github.com/uta-smile/TVT/blob/main/image.png">
</p>


### Datasets:

- Digit: MNIST, SVHN, USPS

- Object: Office, Office-Home, [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

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
Code of ViT is largely borrowed from [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
