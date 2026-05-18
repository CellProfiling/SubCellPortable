"""Utilities for channel-aware automatic inference planning."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


AUTO_MODEL_CHANNELS = "auto"
MANUAL_MODEL_CHANNELS = ("rybg", "rbg", "ybg", "bg")
SUPPORTED_MODEL_CHANNELS = (AUTO_MODEL_CHANNELS, *MANUAL_MODEL_CHANNELS)

STANDARD_IMAGE_COLUMNS = ("r_image", "y_image", "b_image", "g_image")
CHANNEL_TO_COLUMN = {
    "r": "r_image",
    "y": "y_image",
    "b": "b_image",
    "g": "g_image",
}


@dataclass(frozen=True)
class InferencePassSpec:
    """A single inference pass derived from one CSV row."""

    row_index: int
    pass_index: int
    total_passes: int
    model_channels: str
    output_prefix: str
    output_folder: Optional[str]
    channel_paths: Dict[str, str]
    g_column: str
    original_item: Dict[str, Any]


@dataclass(frozen=True)
class PlannedSample:
    """Planned execution for one CSV row."""

    row_index: int
    output_prefix: str
    output_folder: Optional[str]
    original_item: Dict[str, Any]
    pass_specs: tuple[InferencePassSpec, ...]

    @property
    def pass_count(self) -> int:
        return len(self.pass_specs)


@dataclass(frozen=True)
class PlannedInference:
    """Full automatic inference plan for an input CSV."""

    uses_old_format: bool
    g_columns: tuple[str, ...]
    samples: tuple[PlannedSample, ...]

    @property
    def pass_specs(self) -> list[InferencePassSpec]:
        return [pass_spec for sample in self.samples for pass_spec in sample.pass_specs]

    @property
    def pass_count(self) -> int:
        if not self.samples:
            return 0
        return self.samples[0].pass_count

    @property
    def has_multi_pass(self) -> bool:
        return self.pass_count > 1


@dataclass(frozen=True)
class InferencePassResult:
    """Inference result for a single planned pass."""

    pass_spec: InferencePassSpec
    embedding: np.ndarray
    probabilities: Optional[np.ndarray]
    attention_map: Optional[torch.Tensor] = None
    attention_input_shape: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class AggregatedInferenceResult:
    """Final per-sample inference result after pass aggregation."""

    row_index: int
    output_prefix: str
    output_folder: Optional[str]
    embedding: np.ndarray
    probabilities: Optional[np.ndarray]
    pass_count: int
    original_item: Dict[str, Any]
    attention_map: Optional[torch.Tensor] = None
    attention_input_shape: Optional[tuple[int, int]] = None


def normalize_optional_path(value: Any) -> Optional[str]:
    """Normalize CSV cell content to a usable path or None."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return normalized

    return str(value)


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized_record = dict(record)
    for key, value in record.items():
        if key.endswith("_image") or key in {"output_prefix", "output_folder"}:
            normalized_record[key] = normalize_optional_path(value)
    return normalized_record


def _get_g_like_columns(columns: Iterable[str]) -> tuple[str, ...]:
    ordered_columns = list(columns)
    g_columns = []

    if "g_image" in ordered_columns:
        g_columns.append("g_image")

    extra_g_columns = [
        column
        for column in ordered_columns
        if column.endswith("_image") and column not in STANDARD_IMAGE_COLUMNS
    ]
    g_columns.extend(extra_g_columns)
    return tuple(g_columns)


def _select_model_channels(record: Dict[str, Any], g_column: str) -> str:
    if not record.get("b_image"):
        raise ValueError(
            "Automatic model selection requires a non-empty b_image column."
        )

    if not record.get(g_column):
        raise ValueError(
            f"Automatic model selection requires a non-empty {g_column} value for each pass."
        )

    has_r = bool(record.get("r_image"))
    has_y = bool(record.get("y_image"))
    if has_r and has_y:
        return "rybg"
    if has_r:
        return "rbg"
    if has_y:
        return "ybg"
    return "bg"


def _build_channel_paths(
    record: Dict[str, Any], g_column: str, model_channels: str
) -> Dict[str, str]:
    channel_paths = {
        "b": record["b_image"],
        "g": record[g_column],
    }
    if "r" in model_channels:
        channel_paths["r"] = record["r_image"]
    if "y" in model_channels:
        channel_paths["y"] = record["y_image"]
    return channel_paths


def build_auto_inference_plan(csv_path: str) -> PlannedInference:
    """Create an automatic, channel-aware inference plan from the input CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lstrip("#")

    uses_old_format = "output_folder" in df.columns
    g_columns = _get_g_like_columns(df.columns)

    if not g_columns:
        raise ValueError(
            "Automatic model selection requires a g_image column or at least one extra *_image column."
        )

    samples = []
    expected_pass_count = None

    for row_index, raw_record in enumerate(df.to_dict("records")):
        record = _normalize_record(raw_record)
        output_prefix = record.get("output_prefix")
        if not output_prefix:
            raise ValueError(f"Row {row_index + 1} is missing output_prefix.")

        output_folder = record.get("output_folder") if uses_old_format else None
        active_g_columns = [column for column in g_columns if record.get(column)]

        if not active_g_columns:
            raise ValueError(
                f"Row {row_index + 1} ({output_prefix}) does not contain any usable g-like image column."
            )

        if expected_pass_count is None:
            expected_pass_count = len(active_g_columns)
        elif len(active_g_columns) != expected_pass_count:
            raise ValueError(
                "Automatic multi-pass inference requires the same number of g-like image columns per row. "
                f"Row {row_index + 1} ({output_prefix}) has {len(active_g_columns)} passes, "
                f"expected {expected_pass_count}."
            )

        pass_specs = []
        for pass_index, g_column in enumerate(active_g_columns):
            model_channels = _select_model_channels(record, g_column)
            pass_specs.append(
                InferencePassSpec(
                    row_index=row_index,
                    pass_index=pass_index,
                    total_passes=len(active_g_columns),
                    model_channels=model_channels,
                    output_prefix=output_prefix,
                    output_folder=output_folder,
                    channel_paths=_build_channel_paths(
                        record, g_column, model_channels
                    ),
                    g_column=g_column,
                    original_item=record,
                )
            )

        samples.append(
            PlannedSample(
                row_index=row_index,
                output_prefix=output_prefix,
                output_folder=output_folder,
                original_item=record,
                pass_specs=tuple(pass_specs),
            )
        )

    return PlannedInference(
        uses_old_format=uses_old_format,
        g_columns=g_columns,
        samples=tuple(samples),
    )


def group_pass_specs_by_model(
    plan: PlannedInference,
) -> dict[str, list[InferencePassSpec]]:
    """Group pass specs by the model combination they require."""
    grouped: dict[str, list[InferencePassSpec]] = {}
    for pass_spec in plan.pass_specs:
        grouped.setdefault(pass_spec.model_channels, []).append(pass_spec)
    return grouped


def aggregate_auto_results(
    plan: PlannedInference,
    pass_results: Iterable[InferencePassResult],
    embeddings_only: bool,
) -> list[AggregatedInferenceResult]:
    """Aggregate sequential pass results back to one result per CSV row."""
    results_by_key: dict[tuple[int, int], InferencePassResult] = {}

    for pass_result in pass_results:
        key = (pass_result.pass_spec.row_index, pass_result.pass_spec.pass_index)
        if key in results_by_key:
            raise ValueError(
                f"Duplicate pass result for row {pass_result.pass_spec.row_index + 1}, pass {pass_result.pass_spec.pass_index + 1}."
            )
        results_by_key[key] = pass_result

    aggregated_results = []
    for sample in plan.samples:
        sample_pass_results = []
        for pass_spec in sample.pass_specs:
            key = (pass_spec.row_index, pass_spec.pass_index)
            if key not in results_by_key:
                raise ValueError(
                    f"Missing pass result for row {pass_spec.row_index + 1}, pass {pass_spec.pass_index + 1}."
                )
            sample_pass_results.append(results_by_key[key])

        embedding = np.concatenate(
            [pass_result.embedding for pass_result in sample_pass_results], axis=0
        )

        probabilities = None
        if not embeddings_only and sample.pass_count == 1:
            probabilities = sample_pass_results[0].probabilities

        attention_map = None
        attention_input_shape = None
        attention_maps = [
            pass_result.attention_map
            for pass_result in sample_pass_results
            if pass_result.attention_map is not None
        ]
        if attention_maps:
            if len(attention_maps) != len(sample_pass_results):
                raise ValueError(
                    f"Missing attention map for one or more passes in row {sample.row_index + 1}."
                )

            attention_shapes = {tuple(attn.shape) for attn in attention_maps}
            if len(attention_shapes) != 1:
                raise ValueError(
                    f"Incompatible attention map shapes for row {sample.row_index + 1}: {sorted(attention_shapes)}"
                )

            if any(
                pass_result.attention_input_shape is None
                for pass_result in sample_pass_results
            ):
                raise ValueError(
                    f"Missing attention input shape for one or more passes in row {sample.row_index + 1}."
                )
            input_shapes = {
                pass_result.attention_input_shape for pass_result in sample_pass_results
            }
            if len(input_shapes) != 1:
                raise ValueError(
                    f"Incompatible attention input shapes for row {sample.row_index + 1}: {sorted(input_shapes)}"
                )

            attention_map = torch.stack(attention_maps, dim=0).mean(dim=0)
            attention_input_shape = next(iter(input_shapes))

        aggregated_results.append(
            AggregatedInferenceResult(
                row_index=sample.row_index,
                output_prefix=sample.output_prefix,
                output_folder=sample.output_folder,
                embedding=embedding,
                probabilities=probabilities,
                pass_count=sample.pass_count,
                original_item=sample.original_item,
                attention_map=attention_map,
                attention_input_shape=attention_input_shape,
            )
        )

    return aggregated_results
