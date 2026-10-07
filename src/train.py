import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from data_loader import ReconDataset
from models.amr_transformer import amrTransformer

def train(args):
    lr = args.learning_rate
    weight_decay = args.weight_decay
    dropout_rate = args.dropout_rate
    epochs = args.epochs
    batch_size = args.batch_size
    recon_data_root = args.recon_data_root
    output_save_path = args.output_save_path
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    max_files = args.max_files

    train_dataset = ReconDataset(
        hdf5_file_path=recon_data_root,
        context_length=4,
        prediction_length=5,
        waypoint_stride=1,
        image_size=(128, 128),
        goal_min_offset=5,
        goal_max_offset=30,
        split='train',
        train_val_ratio=0.8,
        device=device,
        max_files=max_files
    )

    val_dataset = ReconDataset(
        hdf5_file_path=recon_data_root,
        context_length=4,
        prediction_length=5,
        waypoint_stride=1,
        image_size=(128, 128),
        goal_min_offset=5,
        goal_max_offset=30,
        split='val',
        train_val_ratio=0.8,
        device=device,
        max_files=max_files
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = amrTransformer(embed_dim=128, num_waypoints=5).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs(output_save_path, exist_ok=True)

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            observations = batch['observations'].to(device)
            goals = batch['goals'].to(device)
            waypoints = batch['waypoints'].to(device)

            optimizer.zero_grad()
            predictions = model(observations, goals)
            loss = criterion(predictions, waypoints)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                observations = batch['observations'].to(device)
                goals = batch['goals'].to(device)
                waypoints = batch['waypoints'].to(device)

                predictions = model(observations, goals)
                loss = criterion(predictions, waypoints)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(output_save_path, 'best_model.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best model to {checkpoint_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation using 2D Images and 3D Point Clouds')
    parser.add_argument('--recon_data_root', type=str, default='/home/gkmunda/study/vint/data/recon_dataset/recon_release', help='Path to the NuScenes dataset root')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (cuda or cpu)')
    # parser.add_argument('--input_dim', type=int, default=448, help='Input dimension for the MLP head')
    # parser.add_argument('--output_dim', type=int, default=16, help='Output dimension for the MLP head')
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for the MLP head')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--output_save_path', type=str, default='/home/gkmunda/study/multimodal_project/src/models/checkpoints', help='Path to save the trained model and features')
    parser.add_argument('--max_files', type=int, default=None, help='Max number of HDF5 files to use (for quick testing)')

    parser.add_argument('--dino_model_name', type=str, default='dinov2_vits14', help='DINOv2 model name')
    args = parser.parse_args()

    train(args)