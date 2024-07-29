import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from augmentation import Data_Augmentation

class HumanDataset(Dataset):
    def __init__(self, root_path, aug_img_path, aug_mask_path, data_path, masks_path, num_augmentations, test=False):
        Data_Augmentation(aug_img_path, aug_mask_path, data_path, masks_path, num_augmentations)
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/manual_test/"+i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path+"/manual_test_masks/"+i for i in os.listdir(root_path+"/manual_test_masks/")])
        else:
            self.images = sorted([aug_img_path + "/" + i for i in os.listdir(aug_img_path + "/")])
            self.masks = sorted([aug_mask_path+ "/" + i for i in os.listdir(aug_mask_path+ "/")])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
