import torch
import torchvision
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, image_ids, labels, class_list, crop, transform):
        self.image_ids = image_ids
        self.paths = paths
        self.labels = labels
        self.crop = crop
        if transform:
            self.transforms = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, hue=0.3, saturation=0.3
                    ),
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
                    # transforms.RandomHorizontalFlip(p=0.3),
                    # transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
        self.class_map = {}
        for i, class_ in enumerate(class_list):
            self.class_map[class_] = i

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.paths[self.image_ids[index]])
        image = image.float()
        image = transforms.functional.crop(
            image,
            self.crop["top"],
            self.crop["left"],
            self.crop["height"],
            self.crop["width"],
        )
        image = self.transforms(image)
        label = self.labels[self.image_ids[index]]
        label = self.class_map[label]
        return image, label
