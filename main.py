import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from unet import UNet
from Human_dataset import HumanDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--model_save_path", type=str, default="./models/unet.pth")

    parser.add_argument("--aug_img_path", type=str, default="./data/new_train")
    parser.add_argument("--aug_mask_path", type=str, default="./data/new_train_masks")
    parser.add_argument("--train_path", type=str, default="./data/train")
    parser.add_argument("--masks_path", type=str, default="./data/train_masks")
    parser.add_argument("--num_augmentations", type=int, default=5)
    args = parser.parse_args()

    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DATA_PATH = args.data_path
    MODEL_SAVE_PATH = args.model_save_path

    AUG_IMG_PATH = args.aug_img_path
    AUG_MASK_PATH = args.aug_mask_path
    TRAIN_PATH = args.data_path
    MASKS_PATH = args.masks_path
    NUM_AUGMENTATIONS = args.num_aug

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    train_dataset = HumanDataset(DATA_PATH, AUG_IMG_PATH, AUG_MASK_PATH, TRAIN_PATH, MASKS_PATH, NUM_AUGMENTATIONS)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
