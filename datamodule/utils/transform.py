import torch
import torchvision.transforms as transforms


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, imgs):
        # transform a list of PIL images to Torch.Tensor([T,C,H,W])
        imgs_tensor = torch.stack([self.to_tensor(img) for img in imgs])
        return imgs_tensor
