import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, type, transform=None, label_type=torch.float32):
        if type == 'train':
            self.root = './data/train_preprocessed'
        elif type == 'val':
            self.root = './data/val_preprocessed'
        elif type == 'test':
            self.root = './data/test_preprocessed'
        else:
            raise ValueError('type must be one of [train, val, test]')
        self.transform = transform
        self.image_list = self._get_images_list()
        self.label_type = label_type

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path, label = self.image_list[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.as_tensor(image)
        label = torch.as_tensor(label, dtype=self.label_type)
        return image, label
    
    def _get_images_list(self):
        # non ai images
        nonAIPath = self.root + '/images/'
        fol = os.listdir(nonAIPath)
        neg_result = [[(nonAIPath + x), 0] for x in fol]

        # ai images
        AIPath = self.root + '/images_ai/'
        fol = os.listdir(AIPath)
        pos_result = [[(AIPath + x), 1] for x in fol]

        return neg_result + pos_result