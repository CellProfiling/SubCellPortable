from typing import Dict, Any, List
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import image_utils


def min_max_norm_fn(x: np.ndarray) -> np.ndarray:
    """Normalize array using min-max normalization."""
    min_vals = np.amin(x, keepdims=True)
    max_vals = np.amax(x, keepdims=True)
    return (x - min_vals) / (max_vals - min_vals + 1e-8)


class SubCellDataset(Dataset):
    """PyTorch Dataset for SubCell image processing"""

    def __init__(self, path_list_file=None, model_channels="rybg", data_list=None):
        """
        Args:
            path_list_file (str): Path to the CSV file containing image paths
            model_channels (str): Channel configuration (rybg, rbg, ybg, bg)
            data_list (list): Optional pre-planned rows for automatic inference
        """
        self.model_channels = model_channels
        self.data_list = []
        self.uses_old_format = False

        # Define channel mapping
        self.channel_mapping = {
            "r": "r_image",
            "y": "y_image",
            "b": "b_image",
            "g": "g_image",
        }

        if data_list is not None:
            self.data_list = data_list
            self.uses_old_format = any(
                item.get("output_folder") is not None for item in self.data_list
            )
        else:
            # Read CSV
            df = pd.read_csv(path_list_file)
            df.columns = df.columns.str.lstrip("#")
            self.uses_old_format = "output_folder" in df.columns
            self.data_list = df.to_dict("records")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and preprocess a single image set"""
        item = self.data_list[idx]
        model_channels = item.get("model_channels", self.model_channels)
        channel_paths = item.get("channel_paths")

        # Load images based on model channels configuration
        cell_data = []

        # Only process channels specified in model_channels
        for channel_name in model_channels:
            if channel_paths is not None:
                img_path = channel_paths[channel_name]
            else:
                channel_key = self.channel_mapping[channel_name]
                img_path = item[channel_key]

            img = image_utils.read_grayscale_image(img_path)
            cell_data.append(img)

        # Stack images along channel dimension
        cell_data = np.stack(cell_data, axis=0)  # Shape: (channels, height, width)

        cell_data = min_max_norm_fn(
            cell_data
        )  # (always normalized to 0-1 range as required by model)

        result = {
            "images": cell_data.astype(np.float32),
            "output_prefix": item["output_prefix"],
            "original_item": item.get("original_item", item),
        }

        # Include output_folder only if using old format
        if self.uses_old_format:
            result["output_folder"] = item["output_folder"]

        if "pass_spec" in item:
            result["pass_spec"] = item["pass_spec"]

        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching.

    Args:
        batch: List of dataset items

    Returns:
        Dictionary with batched tensors and lists
    """
    # Stack numpy arrays first, then convert to tensor
    images_np = np.stack([item["images"] for item in batch])
    images = torch.from_numpy(images_np)

    result = {
        "images": images,
        "output_prefixes": [item["output_prefix"] for item in batch],
        "original_items": [item["original_item"] for item in batch],
    }

    # Include output_folders only if present (old format)
    if "output_folder" in batch[0]:
        result["output_folders"] = [item["output_folder"] for item in batch]

    if "pass_spec" in batch[0]:
        result["pass_specs"] = [item["pass_spec"] for item in batch]

    return result
