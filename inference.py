from typing import Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imsave
from torchvision.utils import make_grid
import concurrent.futures
import os

CLASS2NAME = {
    0: "Actin filaments",
    1: "Aggresome",
    2: "Cell Junctions",
    3: "Centriolar satellite",
    4: "Centrosome",
    5: "Cytokinetic bridge",
    6: "Cytoplasmic bodies",
    7: "Cytosol",
    8: "Endoplasmic reticulum",
    9: "Endosomes",
    10: "Focal adhesion sites",
    11: "Golgi apparatus",
    12: "Intermediate filaments",
    13: "Lipid droplets",
    14: "Lysosomes",
    15: "Microtubules",
    16: "Midbody",
    17: "Mitochondria",
    18: "Mitotic chromosome",
    19: "Mitotic spindle",
    20: "Nuclear bodies",
    21: "Nuclear membrane",
    22: "Nuclear speckles",
    23: "Nucleoli",
    24: "Nucleoli fibrillar center",
    25: "Nucleoli rim",
    26: "Nucleoplasm",
    27: "Peroxisomes",
    28: "Plasma membrane",
    29: "Vesicles",
    30: "Negative",
}

CLASS2COLOR = {
    0: "#ffeb3b",
    1: "#76ff03",
    2: "#ff6d00",
    3: "#eb30c1",
    4: "#faadd4",
    5: "#795548",
    6: "#64ffda",
    7: "#00e676",
    8: "#03a9f4",
    9: "#4caf50",
    10: "#ffc107",
    11: "#00bcd4",
    12: "#cddc39",
    13: "#212121",
    14: "#8bc34a",
    15: "#ff9800",
    16: "#ae8c08",
    17: "#ffff00",
    18: "#31b61f",
    19: "#9e9e9e",
    20: "#2196f3",
    21: "#e91e63",
    22: "#3f51b5",
    23: "#9c27b0",
    24: "#673ab7",
    25: "#d3a50b",
    26: "#f44336",
    27: "#009688",
    28: "#ff9e80",
    29: "#242e4b",
    30: "#000000",
}


def save_attention_map(
    attn: torch.Tensor, input_shape: Tuple[int, int], output_path: str
) -> None:
    """Save attention map as PNG image.

    Args:
        attn: Attention tensor
        input_shape: Target (height, width) for interpolation
        output_path: Path prefix for saving (will append _attention_map.png)
    """
    attn = F.interpolate(attn, size=input_shape, mode="bilinear", align_corners=False)
    attn = make_grid(
        attn.permute(1, 0, 2, 3),
        normalize=True,
        nrow=attn.shape[1],
        padding=0,
        scale_each=True,
    )
    attn = (attn.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imsave(output_path + "_attention_map.png", attn)


def _save_single_result(
    output_path: str,
    embedding: np.ndarray,
    probabilities: Optional[np.ndarray],
    embeddings_only: bool,
    save_attention_map_flag: bool,
    attention_map: Optional[torch.Tensor],
    batch_data_shape: Tuple[int, ...],
) -> None:
    """Save results for a single image.

    Args:
        output_path: Path prefix for output files
        embedding: Embedding vector to save
        probabilities: Probability vector (None if embeddings_only)
        embeddings_only: Whether to skip saving probabilities
        save_attention_map_flag: Whether to save attention map
        attention_map: Attention map tensor (single image, with batch dim)
        batch_data_shape: Shape of original batch for attention map resizing
    """
    # Save embedding
    np.save(output_path + "_embedding.npy", embedding)

    # Save probability (only if not embeddings_only)
    if not embeddings_only and probabilities is not None:
        np.save(output_path + "_probabilities.npy", probabilities)

    # Save attention map (configurable)
    if save_attention_map_flag and attention_map is not None:
        save_attention_map(
            attention_map, (batch_data_shape[2], batch_data_shape[3]), output_path
        )


@torch.no_grad()
def run_model(
    model: Any,
    batch_data: torch.Tensor,
    device: torch.device,
    output_paths: List[str],
    save_attention_maps: bool = True,
    embeddings_only: bool = False,
    output_format: str = "individual",
    async_saving: bool = False,
    return_attention_maps: bool = False,
) -> Any:
    """
    Run model inference on a batch of images

    Args:
        model: The ViT model
        batch_data: Tensor of shape (batch_size, channels, height, width)
        device: Device to run inference on
        output_paths: List of output paths for each image in batch
        save_attention_maps: Whether to save attention map images (default: True)
        embeddings_only: If True, skip probability computation and saves (default: False)
        output_format: "individual" for separate files, "h5ad" for combined (default: "individual")
        async_saving: If True, save files asynchronously (default: False)
        return_attention_maps: If True, include attention tensors in combined-mode results

    Returns:
        If async_saving=False: List of (embedding, probabilities) tuples for each image
        If async_saving=True: (results_list, save_future) where save_future can be awaited
        Note: probabilities will be None if embeddings_only=True
    """
    if return_attention_maps and output_format != "combined":
        raise ValueError(
            "return_attention_maps is only supported when output_format='combined'."
        )

    batch_data = batch_data.to(device)
    # Note: Images are already normalized in dataset.py with minmax_norm=True

    # Run model on entire batch
    output = model(batch_data)

    # Convert to numpy once for all items in batch
    embeddings_batch = output.pool_op.cpu().numpy()

    # Only compute probabilities if not in embeddings_only mode
    probabilities_batch = None
    if not embeddings_only:
        # Apply softmax to normalize probabilities (so they sum to 1.0)
        probabilities_batch = F.softmax(output.probabilities, dim=-1).cpu().numpy()

    # Extract attention maps if needed (before freeing GPU memory)
    attention_maps = None
    if save_attention_maps and output.pool_attn is not None:
        attention_maps = output.pool_attn.cpu()  # Move to CPU for saving

    # Free GPU memory immediately - critical for async mode
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare results
    results = []
    input_shape = (batch_data.shape[2], batch_data.shape[3])
    for i, output_path in enumerate(output_paths):
        embedding = embeddings_batch[i]
        probabilities = (
            probabilities_batch[i] if probabilities_batch is not None else None
        )
        if return_attention_maps:
            attention_map = (
                attention_maps[i : i + 1] if attention_maps is not None else None
            )
            results.append((embedding, probabilities, attention_map, input_shape))
        else:
            results.append((embedding, probabilities))

    # Handle file saving based on mode
    if output_format == "combined":
        # For combined format, return results without saving - will be collected and saved at end
        return results

    elif async_saving:
        # Return immediately, save asynchronously
        save_future = _async_save_batch(
            results,
            output_paths,
            embeddings_only,
            save_attention_maps,
            attention_maps,
            batch_data,
            output_format,
        )
        return results, save_future
    else:
        # Synchronous saving (original behavior)
        _sync_save_batch(
            results,
            output_paths,
            embeddings_only,
            save_attention_maps,
            attention_maps,
            batch_data,
            output_format,
        )
        return results


def _sync_save_batch(
    results,
    output_paths,
    embeddings_only,
    save_attention_maps,
    attention_maps,
    batch_data,
    output_format,
):
    """Synchronous batch saving (original behavior)"""
    for i, output_path in enumerate(output_paths):
        embedding, probabilities = results[i]
        attention_map = (
            attention_maps[i : i + 1] if attention_maps is not None else None
        )

        _save_single_result(
            output_path,
            embedding,
            probabilities,
            embeddings_only,
            save_attention_maps,
            attention_map,
            batch_data.shape,
        )


def _async_save_batch(
    results,
    output_paths,
    embeddings_only,
    save_attention_maps,
    attention_maps,
    batch_data,
    output_format,
):
    """Asynchronous batch saving using ThreadPoolExecutor

    Returns:
        Tuple of (executor, futures) - caller should wait on futures and shutdown executor
    """
    # Individual files - can parallelize
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def save_single_item(i):
        output_path = output_paths[i]
        embedding, probabilities = results[i]
        attention_map = (
            attention_maps[i : i + 1] if attention_maps is not None else None
        )

        _save_single_result(
            output_path,
            embedding,
            probabilities,
            embeddings_only,
            save_attention_maps,
            attention_map,
            batch_data.shape,
        )

    # Submit all save tasks and return futures
    futures = [executor.submit(save_single_item, i) for i in range(len(output_paths))]

    return executor, futures
