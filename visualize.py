import typing
import io
import os
import argparse

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
from data.data_list_image import Normalize


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def postprocess_activations(activations):
    output = activations
    output *= 255
    return 255 - output.astype('uint8')


def apply_heatmap(weights, img):
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    return heatmap


def get_heatmaps(activations, img):
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap(weights, img)
    return heatmap


def visualize(args):
    # Prepare Model
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=args.num_classes, 
                                zero_head=False, img_size=args.img_size, vis=True)
    
    model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    
    ad_net = AdversarialNetwork(config.hidden_size//12, config.hidden_size//12)
    ad_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
    ad_net.load_state_dict(torch.load(ad_checkpoint))
    ad_net.eval()
    
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
    ])

    image_list = open(args.image_path).readlines()
    len_ = len(image_list)
    images = [image_list[i].strip().split(" ")[0][1:] for i in range(len_)]

    for image_path in images:
        print(image_path)
        im = rgb_loader(image_path)
        x = transform(im)
        x.size()

        # TVT
        _, att_mat, trans_mat = model(x.unsqueeze(0), ad_net=ad_net)
        
        att_mat = torch.stack(att_mat).squeeze(1)

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        # Attention from the output token to the input space.
        v = joint_attentions[-1]

        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result = get_heatmaps(mask, np.asarray(im))
        
        _ = plt.imshow(result)
        plt.axis('off')

        save_name = "att_" + '_'.join(image_path.split('/')[-2:])
        save_path = os.path.join(args.save_dir, args.dataset, args.name)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="svhn2mnist",
                        help="Which downstream task.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--num_classes", default=10, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--image_path", help="Path of the test image.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--save_dir", default="attention_visual", type=str,
                        help="The directory where attention maps will be saved.")
    args = parser.parse_args()
    visualize(args)

if __name__ == "__main__":
    main()                        