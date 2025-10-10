import os
import shutil
import random
from pathlib import Path
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(source_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path, split_ratio: tuple = (0.7, 0.15, 0.15)):
    """
    Splits image data from a source directory into training, validation, and test sets.

    Args:
        source_dir (Path): Path to the source directory containing class subdirectories.
        train_dir (Path): Path to the training directory.
        val_dir (Path): Path to the validation directory.
        test_dir (Path): Path to the test directory.
        split_ratio (tuple): A tuple with split ratios for train, val, and test.
    """
    try:
        if not source_dir.is_dir():
            logging.error(f"Source directory '{source_dir}' not found or is not a directory.")
            return

        # Create destination directories
        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Get class names from subdirectories
        classes = [d.name for d in source_dir.iterdir() if d.is_dir()]
        if not classes:
            logging.warning(f"No subdirectories (classes) found in '{source_dir}'.")
            return

        logging.info(f"Found classes: {classes}")

        for cls in classes:
            logging.info(f"Processing class: {cls}")
            src_cls_path = source_dir / cls

            # Create class subdirectories in destination folders
            train_cls_path = train_dir / cls
            val_cls_path = val_dir / cls
            test_cls_path = test_dir / cls
            train_cls_path.mkdir(exist_ok=True)
            val_cls_path.mkdir(exist_ok=True)
            test_cls_path.mkdir(exist_ok=True)

            # Get and shuffle all image files
            all_files = [f for f in src_cls_path.iterdir() if f.is_file()]
            random.shuffle(all_files)

            # Calculate split indices
            total_files = len(all_files)
            if total_files == 0:
                logging.warning(f"No files found in class directory: {src_cls_path}")
                continue

            train_split_idx = int(total_files * split_ratio[0])
            val_split_idx = int(total_files * (split_ratio[0] + split_ratio[1]))

            # Get file lists for each set
            train_files = all_files[:train_split_idx]
            val_files = all_files[train_split_idx:val_split_idx]
            test_files = all_files[val_split_idx:]

            # Copy files
            def copy_files(files: list[Path], dest_path: Path):
                for file_path in files:
                    shutil.copy(file_path, dest_path / file_path.name)

            copy_files(train_files, train_cls_path)
            copy_files(val_files, val_cls_path)
            copy_files(test_files, test_cls_path)

            logging.info(f"  - Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

        logging.info("Data splitting complete! ✨")

    except Exception as e:
        logging.error(f"An error occurred during data splitting: {e}", exc_info=True)


if __name__ == '__main__':
    # --- Configuration ---
    SOURCE_DIRECTORY = Path('trashnet_dataset')  # Source directory with raw class folders
    BASE_DEST_DIRECTORY = Path('data')          # Destination for split data

    TRAIN_PATH = BASE_DEST_DIRECTORY / 'train'
    VALIDATION_PATH = BASE_DEST_DIRECTORY / 'validation'
    TEST_PATH = BASE_DEST_DIRECTORY / 'test'

    # Define the split ratio
    SPLIT_RATIO = (0.7, 0.15, 0.15)

    # Run the data splitting function
    split_data(SOURCE_DIRECTORY, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, split_ratio=SPLIT_RATIO)
