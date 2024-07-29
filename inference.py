import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse

from Human_dataset import HumanDataset
from unet import UNet

# def pred_show_image_grid(data_path, model_pth, device):
#     model = UNet(in_channels=3, num_classes=1).to(device)
#     model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
#     image_dataset = HumanDataset(data_path, test=True)
#     images = []
#     orig_masks = []
#     pred_masks = []

#     for img, orig_mask in image_dataset:
#         img = img.float().to(device)
#         img = img.unsqueeze(0)

#         pred_mask = model(img)

#         img = img.squeeze(0).cpu().detach()
#         img = img.permute(1, 2, 0)

#         pred_mask = pred_mask.squeeze(0).cpu().detach()
#         pred_mask = pred_mask.permute(1, 2, 0)
#         pred_mask[pred_mask < 0]=0
#         pred_mask[pred_mask > 0]=1

#         orig_mask = orig_mask.cpu().detach()
#         orig_mask = orig_mask.permute(1, 2, 0)

#         images.append(img)
#         orig_masks.append(orig_mask)
#         pred_masks.append(pred_mask)

#     images.extend(orig_masks)
#     images.extend(pred_masks)
#     fig = plt.figure()
#     for i in range(1, 3*len(image_dataset)+1):
#        fig.add_subplot(3, len(image_dataset), i)
#        plt.imshow(images[i-1], cmap="gray")
#     plt.show()


def single_image_inference(image_pth, model_pth, save_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    # save mask using PIL
    pred_mask = transforms.ToPILImage()(pred_mask)
    pred_mask.save(save_pth + "mask.jpg")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--model_path", type=str, default="./models/unet.pth")
    parser.add_argument("--single_img_path", type=str, default="./data/manual_test/0.jpg")
    parser.add_argument("--save_path", type=str, default="./data/manual_test_masks/")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    SINGLE_IMG_PATH = args.single_img_path
    SAVE_PATH = args.save_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, SAVE_PATH, device)
