import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from data_loader import ReconDataset

def train(args):
    lr = args.learning_rate
    weight_decay = args.weight_decay
    dropout_rate = args.dropout_rate
    epochs = args.epochs
    batch_size = args.batch_size
    recon_data_root = args.recon_data_root
    output_save_path = args.output_save_path
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation using 2D Images and 3D Point Clouds')
    parser.add_argument('--recon_data_root', type=str, default='/home/gkmunda/study/vint/data/recon_dataset/recon_release', help='Path to the NuScenes dataset root')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (cuda or cpu)')
    # parser.add_argument('--input_dim', type=int, default=448, help='Input dimension for the MLP head')
    # parser.add_argument('--output_dim', type=int, default=16, help='Output dimension for the MLP head')
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for the MLP head')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--output_save_path', type=str, default='/home/gkmunda/study/multimodal_project/src/models/checkpoints', help='Path to save the trained model and features')

    parser.add_argument('--dino_model_name', type=str, default='dinov2_vits14', help='DINOv2 model name')
    args = parser.parse_args()

    train(args)