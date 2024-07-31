from hydroDL import kPath

import os
import shutil

def delete_crashed_subdirs(parent_dir):
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # List items in the subdirectory
            items = os.listdir(dir_path)
            # If the subdirectory contains only one item, delete it
            if len(items) <= 1:
                print(f"Deleting directory and its contents: {dir_path}")
                shutil.rmtree(dir_path)

# Example usage
parent_directory = os.path.join(kPath.dirVeg, "runs")
delete_crashed_subdirs(parent_directory)
