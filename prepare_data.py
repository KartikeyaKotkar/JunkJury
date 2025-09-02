import os
import shutil
import random


def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits the data from the source directory into training, validation, and test sets.

    Args:
        source_dir (str): Path to the source directory containing class subdirectories.
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validation directory.
        test_dir (str): Path to the test directory.
        split_ratio (tuple): A tuple containing the split ratios for train, val, and test.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Create destination directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Get class names from the subdirectories in the source folder
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        print(f"Processing class: {cls}")
        src_cls_path = os.path.join(source_dir, cls)

        # Create class subdirectories in train, val, and test folders
        train_cls_path = os.path.join(train_dir, cls)
        val_cls_path = os.path.join(val_dir, cls)
        test_cls_path = os.path.join(test_dir, cls)
        os.makedirs(train_cls_path, exist_ok=True)
        os.makedirs(val_cls_path, exist_ok=True)
        os.makedirs(test_cls_path, exist_ok=True)

        # Get list of all images and shuffle it
        all_files = os.listdir(src_cls_path)
        random.shuffle(all_files)

        # Calculate split indices
        total_files = len(all_files)
        train_split_idx = int(total_files * split_ratio[0])
        val_split_idx = int(total_files * (split_ratio[0] + split_ratio[1]))

        # Get the lists of files for each set
        train_files = all_files[:train_split_idx]
        val_files = all_files[train_split_idx:val_split_idx]
        test_files = all_files[val_split_idx:]

        # Function to copy files
        def copy_files(files, dest_path):
            for file_name in files:
                shutil.copy(os.path.join(src_cls_path, file_name), os.path.join(dest_path, file_name))

        # Copy files to their respective directories
        copy_files(train_files, train_cls_path)
        copy_files(val_files, val_cls_path)
        copy_files(test_files, test_cls_path)

        print(f"  - Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

    print("\nData splitting complete! ✨")


if __name__ == '__main__':
    # --- Configuration ---
    SOURCE_DIRECTORY = 'trashnet_dataset'  # The directory with your raw class folders
    BASE_DEST_DIRECTORY = 'data'  # A new folder to hold the split data

    TRAIN_PATH = os.path.join(BASE_DEST_DIRECTORY, 'train')
    VALIDATION_PATH = os.path.join(BASE_DEST_DIRECTORY, 'validation')
    TEST_PATH = os.path.join(BASE_DEST_DIRECTORY, 'test')

    # Run the function
    # We use a 70% training, 15% validation, 15% testing split
    split_data(SOURCE_DIRECTORY, TRAIN_PATH, VALIDATION_PATH, TEST_PATH)