import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
import argparse

def load_images_and_masks(image_dir, mask_dir):
        images = []
        masks = []
        for file_name in os.listdir(image_dir):
            img_name = file_name.split('.')[0]
            mask_name = img_name + '.png'
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image = cv2.imread(os.path.join(image_dir, file_name))
                mask = cv2.imread(os.path.join(mask_dir, mask_name))
                images.append(image)
                masks.append(mask)
        print('Loaded {} images'.format(len(images)))
        return images, masks

def aug_images_and_masks(images, masks, seq, num_augmentations):
    imgs = []
    msks = []
    for i in range(num_augmentations):
        augmented = seq(images=images, segmentation_maps=masks)
        img, mask = augmented
        imgs.append(img)
        msks.append(mask)
    return [img for sublist in imgs for img in sublist], [mask for sublist in msks for mask in sublist]

def save_images_and_masks(images, masks, train_dir, mask_dir):
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join(train_dir, '{}.jpg'.format(i)), image)
    for i, mask in enumerate(masks):
        cv2.imwrite(os.path.join(mask_dir, '{}.png'.format(i)), mask)

def Data_Augmentation(aug_img_path, aug_mask_path, data_path, masks_path, num_augmentations):
    ia.seed(1)

    images, masks = load_images_and_masks(data_path, masks_path)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
]) 
    augmented_images, augmented_masks = aug_images_and_masks(images, masks, seq, num_augmentations)
    save_images_and_masks(augmented_images, augmented_masks, aug_img_path, aug_mask_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/train")
    parser.add_argument("--masks_path", type=str, default="./data/train_masks")
    parser.add_argument("--aug_img_path", type=str, default="./data/aug_images")
    parser.add_argument("--aug_mask_path", type=str, default="./data/aug_masks")
    parser.add_argument("--num_augmentations", type=int, default=5)
    args = parser.parse_args()

    Data_Augmentation(args.aug_img_path, args.aug_mask_path, args.data_path, args.masks_path, args.num_augmentations)