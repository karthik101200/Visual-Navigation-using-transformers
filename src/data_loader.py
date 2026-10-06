import io

from torch.utils.data import Dataset
import os
from pathlib import Path
import sys
import numpy as np
import torch
import h5py
from PIL import Image


class ReconDataset(Dataset):
    def __init__(self, hdf5_file_path,
                 context_length = 4,
                 prediction_length = 1,
                 waypoint_stride = 1,
                 image_size = (128, 128),
                 goal_min_offset = 5,
                 goal_max_offset = 30):

        self.hdf5_file_path = hdf5_file_path
        self.context_length = context_length
        self.waypoint_stride = waypoint_stride
        self.image_size = image_size
        self.goal_min_offset = goal_min_offset
        self.goal_max_offset = goal_max_offset
        self.prediction_length = prediction_length

        self.files = self._load_hdf5_files()

        self.samples = []
        for idx, file in enumerate(self.files):
            # read hdf5 file and get valid timestamps
            file_path = str(file)
            try:
                with h5py.File(file_path, 'r') as f:
                    n = len(f['images']['rgb_left'])
                    for t in range(n):
                        # Check if there are enough previous observations for context
                        if t >= self.context_length - 1 and t + self.prediction_length*self.waypoint_stride < n and t + self.goal_min_offset < n and t + self.goal_max_offset < n:
                            self.samples.append((idx, t))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    def _load_hdf5_files(self):
        # load all hdf5 files in the directory and sort it according to <robot>_<data & time>_<trajectory_index>_<chunk_index>

        def sort_key(path):
            if path.suffix == '.hdf5':
                parts = path.stem.split('_')
                if len(parts) >= 4:
                    robot = parts[0]
                    datetime = parts[1]
                    trajectory_index = int(parts[2])
                    chunk_index = int(parts[3][1:])
                    return (robot, datetime, trajectory_index, chunk_index)
            return (path.suffix != '.hdf5', path.stem)

        files = list(Path(self.hdf5_file_path).glob('*.hdf5'))
        files.sort(key=sort_key)

        return files

    def _decode_rgb_image(self, t, file_path):
        # Decode RGB Image from index t
        with h5py.File(file_path, 'r') as f:
            image_bytes = f["images"]["rgb_left"][t]
            image = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
            image = image.resize(self.image_size)
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            return image 
                
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        file_idx, t = self.samples[idx]
        file_path = str(self.files[file_idx])

        with h5py.File(file_path, 'r') as f:
            T = len(f['images']['rgb_left'])  # Total number of timestamps in the file
            # sample observations as [t - context_length + 1, t - context_length + 2, ..., t]
            observation_indices = []
            for i in range(self.context_length):
                observation_indices.append(t - self.context_length + 1 + i)
           
            # sample goal as [t + goal_offset]
            goal_index = np.random.randint(t+self.goal_min_offset, min(t+self.goal_max_offset+1,T))
            observation_images = []
            goal_images = []
            for obs_index in observation_indices:
                image = self._decode_rgb_image(obs_index, file_path)
                observation_images.append(image)

            goal_images.append(self._decode_rgb_image(goal_index, file_path))
            observation_images = np.stack(observation_images, axis=0)
            goal_images = np.stack(goal_images, axis=0)

            # Get goal position and yaw
            current_position = f['jackal']['position'][t, :2]
            current_yaw = f['jackal']['yaw'][t]

            waypoints = []
            for i in range(self.prediction_length):
                pred_index = t + (i+1)*self.waypoint_stride

                # Read poistion data from the hdf5 file
                pred_position = f['jackal']['position'][pred_index,:2]

                # Convert position and yaw into relative coordinates with respect to the goal
                relative_position = pred_position - current_position 
                rotation_matrix = np.array([[np.cos(current_yaw), np.sin(current_yaw)],
                                            [-np.sin(current_yaw), np.cos(current_yaw)]])

                relative_position = rotation_matrix @ relative_position
                waypoints.append(relative_position)

            waypoints = np.stack(waypoints, axis=0)

            # return tensors for observations, goals, waypoints, current index and goal_index
            return {
                'observations': torch.tensor(observation_images, dtype=torch.float32),
                'goals': torch.tensor(goal_images, dtype=torch.float32),
                'waypoints': torch.tensor(waypoints, dtype=torch.float32),
                'current_index': t,
                'goal_index': goal_index
            }





                



                
