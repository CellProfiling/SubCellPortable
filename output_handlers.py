"""Output handling for different file formats."""

import os
import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import h5py

import inference
from config import NUM_CLASSES, RESULT_CSV_FILE

logger = logging.getLogger(__name__)


def create_csv_columns(has_classifier: bool, embedding_dim: int) -> List[str]:
    """Create column names for CSV output.

    Args:
        has_classifier: Whether classifier predictions are included

    Returns:
        List of column names
    """
    columns = ["id"]

    if has_classifier:
        columns.extend(
            [
                "top_class_name",
                "top_class",
                "top_3_classes_names",
                "top_3_classes",
            ]
        )
        # Probability columns
        columns.extend([f"prob{i:02d}" for i in range(NUM_CLASSES)])

    # Feature columns
    columns.extend([f"feat{i:04d}" for i in range(embedding_dim)])

    return columns


def compute_top_predictions(probabilities: np.ndarray) -> Tuple[int, str, str, str]:
    """Compute top class and top-3 classes from probabilities.

    Args:
        probabilities: Array of class probabilities

    Returns:
        Tuple of (top_class_idx, top_class_name, top_3_names, top_3_indices)
    """
    probs_list = probabilities.tolist()

    # Top class
    top_class = probs_list.index(max(probs_list))
    top_class_name = inference.CLASS2NAME[top_class]

    # Top 3 classes
    top_3_indices = sorted(
        range(len(probs_list)), key=lambda i: probs_list[i], reverse=True
    )[:3]
    top_3_names = ",".join([inference.CLASS2NAME[i] for i in top_3_indices])
    top_3_str = ",".join(map(str, top_3_indices))

    return top_class, top_class_name, top_3_names, top_3_str


def create_csv_row(
    output_prefix: str,
    embedding: np.ndarray,
    probabilities: Optional[np.ndarray],
    has_classifier: bool,
) -> List:
    """Create a single CSV row from results.

    Args:
        output_prefix: ID/prefix for this sample
        embedding: Embedding vector
        probabilities: Probability vector (None if embeddings_only)
        has_classifier: Whether classifier is being used

    Returns:
        List representing CSV row
    """
    row = [output_prefix]

    if has_classifier:
        if probabilities is not None:
            top_class, top_class_name, top_3_names, top_3_str = compute_top_predictions(
                probabilities
            )
            row.extend([top_class_name, top_class, top_3_names, top_3_str])
            row.extend(probabilities.tolist())
        else:
            row.extend(["", np.nan, "", ""])
            row.extend([np.nan] * NUM_CLASSES)

    row.extend(embedding.tolist())
    return row


class CSVOutputHandler:
    """Handler for CSV output format."""

    def __init__(self, has_classifier: bool, embedding_dim: Optional[int] = None):
        """Initialize CSV handler.

        Args:
            has_classifier: Whether classifier predictions are included
        """
        self.has_classifier = has_classifier
        self.embedding_dim = embedding_dim
        self.columns: Optional[List[str]] = None
        self.rows = []

    def _ensure_columns(self, embedding_dim: int) -> None:
        if self.embedding_dim is None:
            self.embedding_dim = embedding_dim
            self.columns = create_csv_columns(self.has_classifier, self.embedding_dim)
            return

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Inconsistent embedding size detected: expected {self.embedding_dim}, got {embedding_dim}."
            )

        if self.columns is None:
            self.columns = create_csv_columns(self.has_classifier, self.embedding_dim)

    def add_batch(
        self,
        output_prefixes: List[str],
        embeddings: List[np.ndarray],
        probabilities_list: List[Optional[np.ndarray]],
    ) -> None:
        """Add a batch of results.

        Args:
            output_prefixes: List of sample IDs
            embeddings: List of embedding vectors
            probabilities_list: List of probability vectors (None entries if embeddings_only)
        """
        for prefix, embedding, probs in zip(
            output_prefixes, embeddings, probabilities_list
        ):
            self._ensure_columns(len(embedding))
            row = create_csv_row(prefix, embedding, probs, self.has_classifier)
            self.rows.append(row)

    def save(self, output_path: str = RESULT_CSV_FILE) -> None:
        """Save accumulated results to CSV.

        Args:
            output_path: Path to save CSV file
        """
        if not self.rows:
            logger.warning("No data to save to CSV")
            return

        if self.columns is None:
            logger.warning("No columns configured for CSV output")
            return

        df = pd.DataFrame(self.rows, columns=self.columns)
        df.to_csv(output_path, index=False)
        logger.info(f"CSV saved to {output_path} with {len(self.rows)} rows")


class H5ADOutputHandler:
    """Handler for H5AD output format."""

    def __init__(self, output_dir: str):
        """Initialize H5AD handler.

        Args:
            output_dir: Directory to save H5AD file
        """
        self.output_dir = output_dir
        self.embeddings = []
        self.probabilities = []
        self.image_names = []
        self.has_any_probabilities = False

    def add_batch(
        self,
        output_prefixes: List[str],
        embeddings: List[np.ndarray],
        probabilities_list: List[Optional[np.ndarray]],
    ) -> None:
        """Add a batch of results.

        Args:
            output_prefixes: List of sample IDs
            embeddings: List of embedding vectors
            probabilities_list: List of probability vectors (None entries if embeddings_only)
        """
        for prefix, embedding, probs in zip(
            output_prefixes, embeddings, probabilities_list
        ):
            self.embeddings.append(embedding)
            if probs is not None:
                self.has_any_probabilities = True
            self.probabilities.append(probs)
            self.image_names.append(prefix)

    def save(self, embeddings_only: bool = False) -> str:
        """Save accumulated results to H5AD file.

        Args:
            embeddings_only: Whether only embeddings were computed

        Returns:
            Path to saved H5AD file
        """
        if not self.embeddings:
            logger.warning("No data to save to H5AD")
            return ""

        logger.info(f"Saving H5AD file with {len(self.embeddings)} embeddings...")

        os.makedirs(self.output_dir, exist_ok=True)
        h5ad_path = os.path.join(self.output_dir, "embeddings.h5ad")

        embeddings_array = np.stack(self.embeddings)

        with h5py.File(h5ad_path, "w") as f:
            # Save embeddings as the main data matrix (AnnData convention)
            f.create_dataset("X", data=embeddings_array)

            # Save observation names (image names)
            obs_names = np.array(self.image_names, dtype="S")
            f.create_dataset("obs/index", data=obs_names)

            # Save probabilities if available
            if self.has_any_probabilities:
                probabilities_array = np.stack(
                    [
                        (
                            probs
                            if probs is not None
                            else np.full(NUM_CLASSES, np.nan, dtype=np.float32)
                        )
                        for probs in self.probabilities
                    ]
                )
                f.create_dataset("obsm/probabilities", data=probabilities_array)

            # Add metadata
            f.attrs["n_obs"] = len(self.embeddings)
            f.attrs["n_vars"] = embeddings_array.shape[1]
            f.attrs["created_by"] = "SubCellPortable"
            f.attrs["embeddings_only"] = embeddings_only

        logger.info(f"H5AD file saved: {h5ad_path}")
        logger.info(f"Shape: {embeddings_array.shape}")
        logger.info(
            f"Contains: embeddings, image_names"
            + (", probabilities" if self.has_any_probabilities else "")
        )

        return h5ad_path
