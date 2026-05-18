"""Main inference orchestration for SubCellPortable."""

import datetime
import logging
import os
import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import inference
from vit_model import ViTPoolClassifier
from dataset import SubCellDataset, collate_fn
from config import SubCellConfig, PATH_LIST_CSV, LOG_FILE
from cli import parse_args
from inference_planner import (
    AUTO_MODEL_CHANNELS,
    InferencePassResult,
    aggregate_auto_results,
    build_auto_inference_plan,
    group_pass_specs_by_model,
)
from model_loader import ensure_models_available
from output_handlers import CSVOutputHandler, H5ADOutputHandler, compute_top_predictions

# Set CUDA device ordering
os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_logging(log_file: str = LOG_FILE) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger
    """
    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(levelname)s: %(message)s",
        filemode="w",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    logger = logging.getLogger("SubCell inference")
    return logger


def load_config() -> tuple[SubCellConfig, str, str]:
    """Load configuration from multiple sources.

    Configuration priority (highest to lowest):
    1. Command-line arguments
    2. config.yaml file (or custom config file)
    3. Default values

    Returns:
        Tuple of (config object, config_file_path, path_list_path)
    """
    # Parse CLI args (only explicitly provided args)
    cli_args = parse_args()

    # Get config file path (default or custom)
    config_file = cli_args.pop("config", "config.yaml")

    # Get path_list file path (default or custom)
    path_list = cli_args.pop("path_list", PATH_LIST_CSV)

    # Start with defaults
    config_dict = {}

    # Load from config file if exists
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict.update(yaml_config)

    # Override with command-line arguments
    # Note: cli_args only contains explicitly provided arguments (not defaults)
    # This ensures config file values aren't overwritten by CLI defaults
    config_dict.update(cli_args)

    # Create config object (with validation)
    return SubCellConfig.from_dict(config_dict), config_file, path_list


def setup_device(gpu_id: int, logger: logging.Logger) -> torch.device:
    """Set up computing device (CPU or GPU).

    Args:
        gpu_id: GPU ID to use (-1 for CPU)
        logger: Logger instance

    Returns:
        Configured torch device
    """
    if torch.cuda.is_available() and gpu_id != -1:
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU: cuda:{gpu_id}")
    else:
        if gpu_id != -1:
            logger.warning(f"GPU {gpu_id} requested but CUDA not available. Using CPU.")
        else:
            logger.info("Using CPU")
        device = torch.device("cpu")
    return device


def create_dataloader(
    csv_path: str,
    config: SubCellConfig,
    logger: logging.Logger,
    model_channels: str = None,
    data_list: list = None,
) -> DataLoader:
    """Create DataLoader for batch processing.

    Args:
        csv_path: Path to CSV file with image paths
        config: Configuration object
        logger: Logger instance

    Returns:
        Configured DataLoader

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if data_list is None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Input file not found: {csv_path}. "
                f"Please create this file with your image paths."
            )

        dataset = SubCellDataset(csv_path, model_channels or config.model_channels)
    else:
        dataset = SubCellDataset(
            model_channels=model_channels or config.model_channels,
            data_list=data_list,
        )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_fn,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

    dataloader = DataLoader(**dataloader_kwargs)

    logger.info(f"Processing {len(dataset)} images in batches of {config.batch_size}")
    logger.info(f"Using {config.num_workers} workers for data loading")

    return dataloader


def process_batch_results(
    batch_results: list,
    output_prefixes: list,
    classifier_paths: list,
    config: SubCellConfig,
) -> None:
    """Process and log batch results.

    Args:
        batch_results: List of (embedding, probabilities) tuples
        output_prefixes: List of output prefixes for logging
        classifier_paths: Classifier paths (None if embeddings_only)
        config: Configuration object
    """
    if config.quiet:
        return

    for i, (embedding, probabilities) in enumerate(batch_results):
        output_prefix = output_prefixes[i]
        log_message = f"- Saved results for {output_prefix}"

        if classifier_paths and probabilities is not None:
            _, _, top_3_names, _ = compute_top_predictions(probabilities)
            log_message += f", locations predicted [{top_3_names}]"

        config.log.info(log_message)


def validate_csv_format(
    uses_old_format: bool,
    config: SubCellConfig,
    logger: logging.Logger,
) -> None:
    """Validate CSV format and output directory requirements."""
    if uses_old_format:
        logger.warning(
            "⚠️  DEPRECATION WARNING: Your path_list.csv uses the old format with 'output_folder' column."
        )
        logger.warning(
            "This format is deprecated and will be removed in a future version."
        )
        logger.warning(
            "Please update to the new format: remove 'output_folder' column and use --output_dir instead."
        )
        logger.warning("See documentation for migration guide.")
    elif not config.output_dir:
        raise ValueError(
            "output_dir is required when using new CSV format (without output_folder column). "
            "Please specify via --output_dir or in config.yaml"
        )


def build_output_paths(
    output_prefixes: list,
    uses_old_format: bool,
    config: SubCellConfig,
    output_folders: list = None,
) -> list:
    """Build output paths for a batch."""
    if uses_old_format:
        for output_folder in set(output_folders):
            os.makedirs(output_folder, exist_ok=True)

        return [
            os.path.join(output_folders[i], output_prefixes[i])
            for i in range(len(output_prefixes))
        ]

    output_paths = []
    for prefix in output_prefixes:
        full_path = os.path.join(config.output_dir, prefix)
        output_dir_for_file = os.path.dirname(full_path)
        if output_dir_for_file:
            os.makedirs(output_dir_for_file, exist_ok=True)
        output_paths.append(full_path)

    return output_paths


def create_planned_items(pass_specs: list) -> list:
    """Convert planned inference passes into dataset rows."""
    data_list = []
    for pass_spec in pass_specs:
        item = {
            "output_prefix": pass_spec.output_prefix,
            "original_item": pass_spec.original_item,
            "model_channels": pass_spec.model_channels,
            "channel_paths": pass_spec.channel_paths,
            "pass_spec": pass_spec,
        }

        if pass_spec.output_folder is not None:
            item["output_folder"] = pass_spec.output_folder

        data_list.append(item)

    return data_list


def load_model(
    model_channels: str,
    config: SubCellConfig,
    embeddings_only: bool,
    device: torch.device,
) -> tuple[ViTPoolClassifier, list]:
    """Load the configured model and classifier weights."""
    classifier_paths, encoder_path, model_config = ensure_models_available(
        model_channels,
        config.model_type,
        embeddings_only,
        config.update_model,
    )

    model = ViTPoolClassifier(model_config)
    classifier_paths_for_loading = (
        classifier_paths if classifier_paths is not None else []
    )
    model.load_model_dict(encoder_path, classifier_paths_for_loading)
    model.eval()
    model.to(device)

    return model, classifier_paths


def save_aggregated_results(
    aggregated_results: list,
    uses_old_format: bool,
    embeddings_only: bool,
    config: SubCellConfig,
) -> None:
    """Save aggregated automatic-inference results."""
    output_prefixes = [result.output_prefix for result in aggregated_results]
    output_folders = None
    if uses_old_format:
        output_folders = [result.output_folder for result in aggregated_results]

    output_paths = build_output_paths(
        output_prefixes,
        uses_old_format,
        config,
        output_folders,
    )

    if config.output_format == "individual":
        for output_path, result in zip(output_paths, aggregated_results):
            np.save(output_path + "_embedding.npy", result.embedding)
            if result.probabilities is not None:
                np.save(output_path + "_probabilities.npy", result.probabilities)

    if config.save_attention_maps:
        for output_path, result in zip(output_paths, aggregated_results):
            if (
                result.attention_map is not None
                and result.attention_input_shape is not None
            ):
                inference.save_attention_map(
                    result.attention_map,
                    result.attention_input_shape,
                    output_path,
                )

    has_classifier = any(
        result.probabilities is not None for result in aggregated_results
    )
    embeddings = [result.embedding for result in aggregated_results]
    probabilities_list = [result.probabilities for result in aggregated_results]

    if config.create_csv:
        csv_handler = CSVOutputHandler(has_classifier=has_classifier)
        csv_handler.add_batch(output_prefixes, embeddings, probabilities_list)
        if uses_old_format or not config.output_dir:
            csv_handler.save("result.csv")
        else:
            csv_handler.save(os.path.join(config.output_dir, "result.csv"))

    if config.output_format == "combined":
        if uses_old_format:
            output_dir = aggregated_results[0].output_folder
        else:
            output_dir = config.output_dir

        h5ad_handler = H5ADOutputHandler(output_dir)
        h5ad_handler.add_batch(output_prefixes, embeddings, probabilities_list)
        h5ad_handler.save(embeddings_only=embeddings_only)

    batch_results = list(zip(embeddings, probabilities_list))
    process_batch_results(
        batch_results,
        output_prefixes,
        [1] if has_classifier else None,
        config,
    )


def run_inference() -> None:
    """Main inference pipeline."""
    # Track overall timing
    start_time = datetime.datetime.now()

    # Setup - load config and get file paths
    config, config_file, path_list = load_config()

    # Override path_list location if specified in config
    if config.path_list:
        path_list = config.path_list

    # Setup logging with output_dir if specified
    if config.output_dir:
        log_path = os.path.join(config.output_dir, "log.txt")
        os.makedirs(config.output_dir, exist_ok=True)
        logger = setup_logging(log_path)
    else:
        logger = setup_logging()

    config.log = logger

    # Log start
    logger.info("=" * 60)
    logger.info("SubCellPortable - Subcellular Localization Inference")
    logger.info("=" * 60)
    logger.info(f"Start: {start_time.strftime('%Y/%m/%d %H:%M:%S')}")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Input CSV: {path_list}")
    logger.info("Parameters:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 60)

    try:
        device = setup_device(config.gpu, logger)
        csv_handler = None
        h5ad_handler = None
        pending_executors = []
        pending_futures = []
        pass_results = []
        total_images = 0
        uses_old_format = False
        effective_embeddings_only = config.embeddings_only
        auto_plan = None

        if config.model_channels == AUTO_MODEL_CHANNELS:
            if config.async_saving:
                logger.warning(
                    "async_saving is ignored with automatic model selection."
                )

            auto_plan = build_auto_inference_plan(path_list)
            uses_old_format = auto_plan.uses_old_format
            validate_csv_format(uses_old_format, config, logger)

            effective_embeddings_only = (
                config.embeddings_only or auto_plan.has_multi_pass
            )
            if auto_plan.has_multi_pass and not config.embeddings_only:
                logger.warning(
                    "Automatic multi-pass inference concatenates embeddings across g-like channels and disables classification output."
                )
            if auto_plan.has_multi_pass and config.save_attention_maps:
                logger.info(
                    "Saving one mean attention map per sample across sequential g-like passes."
                )

            model_groups = group_pass_specs_by_model(auto_plan)
            logger.info("Automatic model selection enabled")
            logger.info(f"Detected g-like columns: {', '.join(auto_plan.g_columns)}")
            logger.info(
                f"Planned {len(auto_plan.samples)} samples across {len(auto_plan.pass_specs)} passes "
                f"({auto_plan.pass_count} pass(es) per sample)."
            )
            for model_channels, pass_specs in model_groups.items():
                logger.info(f"  - {model_channels}: {len(pass_specs)} pass(es)")
        else:
            model_groups = {config.model_channels: None}

        if effective_embeddings_only:
            logger.info("🔍 Running in EMBEDDINGS ONLY mode - no classification")
        else:
            logger.info("🎯 Running in FULL mode - embeddings + classification")

        for model_channels, pass_specs in model_groups.items():
            model, classifier_paths = load_model(
                model_channels,
                config,
                effective_embeddings_only,
                device,
            )

            if pass_specs is None:
                dataloader = create_dataloader(path_list, config, logger)
                uses_old_format = dataloader.dataset.uses_old_format
                validate_csv_format(uses_old_format, config, logger)

                if config.create_csv and csv_handler is None:
                    csv_handler = CSVOutputHandler(
                        has_classifier=classifier_paths is not None
                    )

                if config.output_format == "combined" and h5ad_handler is None:
                    if uses_old_format:
                        first_item = dataloader.dataset.data_list[0]
                        h5ad_handler = H5ADOutputHandler(first_item["output_folder"])
                    else:
                        h5ad_handler = H5ADOutputHandler(config.output_dir)

                total_images = len(dataloader.dataset)
            else:
                dataloader = create_dataloader(
                    None,
                    config,
                    logger,
                    model_channels=model_channels,
                    data_list=create_planned_items(pass_specs),
                )
                logger.info(
                    f"Processing {len(dataloader.dataset)} planned pass(es) with the {model_channels} model in batches of {config.batch_size}"
                )
                total_images = len(auto_plan.samples)

            for batch in tqdm(
                dataloader,
                desc=(
                    f"Processing {model_channels}"
                    if pass_specs is not None
                    else "Processing batches"
                ),
                unit="batch",
            ):
                images = batch["images"]
                output_prefixes = batch["output_prefixes"]
                output_paths = build_output_paths(
                    output_prefixes,
                    uses_old_format,
                    config,
                    batch.get("output_folders"),
                )

                try:
                    inference_result = inference.run_model(
                        model,
                        images,
                        device,
                        output_paths,
                        save_attention_maps=config.save_attention_maps,
                        embeddings_only=effective_embeddings_only,
                        output_format=(
                            "combined"
                            if auto_plan is not None
                            else config.output_format
                        ),
                        async_saving=(
                            False if auto_plan is not None else config.async_saving
                        ),
                        return_attention_maps=config.save_attention_maps
                        and auto_plan is not None,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error("=" * 60)
                        logger.error("❌ CUDA OUT OF MEMORY ERROR")
                        logger.error("=" * 60)
                        logger.error(f"Current configuration:")
                        logger.error(f"  batch_size: {config.batch_size}")
                        logger.error(f"  num_workers: {config.num_workers}")
                        logger.error(f"  GPU: {config.gpu}")
                        logger.error("")
                        logger.error("💡 Suggestions to fix:")
                        logger.error(
                            f"  1. Reduce batch_size: -b {max(1, config.batch_size // 2)}"
                        )
                        logger.error(
                            f"  2. Use fewer workers: -w {max(1, config.num_workers // 2)}"
                        )
                        logger.error(
                            "  3. Switch to CPU: -g -1 (slower but uses RAM instead)"
                        )
                        logger.error("  4. Close other GPU applications")
                        logger.error("=" * 60)
                    raise

                if auto_plan is not None:
                    batch_results = inference_result
                elif config.output_format == "combined":
                    batch_results = inference_result
                elif config.async_saving:
                    batch_results, (executor, futures) = inference_result
                    pending_executors.append(executor)
                    pending_futures.extend(futures)
                else:
                    batch_results = inference_result

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if auto_plan is not None:
                    for pass_spec, batch_result in zip(
                        batch["pass_specs"], batch_results
                    ):
                        if config.save_attention_maps:
                            (
                                embedding,
                                probabilities,
                                attention_map,
                                attention_input_shape,
                            ) = batch_result
                        else:
                            embedding, probabilities = batch_result
                            attention_map = None
                            attention_input_shape = None

                        pass_results.append(
                            InferencePassResult(
                                pass_spec=pass_spec,
                                embedding=embedding,
                                probabilities=probabilities,
                                attention_map=attention_map,
                                attention_input_shape=attention_input_shape,
                            )
                        )
                    continue

                embeddings = [result[0] for result in batch_results]
                probabilities_list = [result[1] for result in batch_results]

                if csv_handler:
                    csv_handler.add_batch(
                        output_prefixes, embeddings, probabilities_list
                    )

                if h5ad_handler:
                    h5ad_handler.add_batch(
                        output_prefixes, embeddings, probabilities_list
                    )

                process_batch_results(
                    batch_results, output_prefixes, classifier_paths, config
                )

            del dataloader
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if auto_plan is not None:
            aggregated_results = aggregate_auto_results(
                auto_plan,
                pass_results,
                embeddings_only=effective_embeddings_only,
            )
            save_aggregated_results(
                aggregated_results,
                uses_old_format,
                effective_embeddings_only,
                config,
            )
        else:
            if config.async_saving and pending_futures:
                logger.info(
                    f"Waiting for {len(pending_futures)} async save operations..."
                )
                import concurrent.futures

                concurrent.futures.wait(pending_futures)
                for executor in pending_executors:
                    executor.shutdown(wait=True)
                logger.info("All async saves completed")

            if csv_handler:
                if uses_old_format or not config.output_dir:
                    csv_handler.save("result.csv")
                else:
                    csv_handler.save(os.path.join(config.output_dir, "result.csv"))

            if h5ad_handler:
                h5ad_handler.save(embeddings_only=config.embeddings_only)

        # Calculate timing statistics
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        images_per_sec = (
            total_images / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        )

        # Log success summary
        logger.info("-" * 60)
        logger.info("✓ Processing completed successfully")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Average speed: {images_per_sec:.2f} images/sec")
        logger.info(
            f"Average time per image: {elapsed.total_seconds()/total_images:.4f} sec"
        )

        # Log output location
        if config.output_dir:
            logger.info(f"Output saved to: {os.path.abspath(config.output_dir)}")
            if config.create_csv:
                logger.info(f"  - result.csv")
            if config.output_format == "combined":
                logger.info(f"  - embeddings.h5ad")
            logger.info(f"  - log.txt")
            if config.output_format == "individual":
                logger.info(f"  - {total_images} embedding files (*_embedding.npy)")
                if not effective_embeddings_only:
                    logger.info(
                        f"  - {total_images} probability files (*_probabilities.npy)"
                    )
                if config.save_attention_maps:
                    logger.info(
                        f"  - {total_images} attention maps (*_attention_map.png)"
                    )

    except FileNotFoundError as e:
        logger.error("-" * 60)
        logger.error(f"✗ File not found: {e}")
        logger.error("Please check that all input files exist.")
        raise
    except ValueError as e:
        logger.error("-" * 60)
        logger.error(f"✗ Configuration error: {e}")
        logger.error("Please check your configuration and CSV format.")
        raise
    except RuntimeError as e:
        logger.error("-" * 60)
        logger.error(f"✗ Runtime error: {e}")
        if "out of memory" in str(e).lower():
            logger.error("Suggestion: Try reducing batch_size in config.yaml")
        raise
    except Exception as e:
        logger.error("-" * 60)
        logger.error(f"✗ Unexpected error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

    finally:
        end_time = datetime.datetime.now()
        logger.info("=" * 60)
        logger.info(f"End: {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
        logger.info("=" * 60)


if __name__ == "__main__":
    run_inference()
