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
    parser.add_argument("--model_save_path", type=str, default="./models")
    parser.add_argument("--load_weight", type=str, default=None)
    args = parser.parse_args()

    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DATA_PATH = args.data_path
    MODEL_SAVE_PATH = args.model_save_path
    LOAD_WEIGHT = args.load_weight

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    train_dataset = HumanDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    if LOAD_WEIGHT:
        print("Loading weights", LOAD_WEIGHT)
        checkpoint = torch.load(LOAD_WEIGHT)
        model.load_state_dict(checkpoint)

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
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.6f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.6f}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH + f"/unet_epoch_{epoch+1}.pth")
        print("Model Saved")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
