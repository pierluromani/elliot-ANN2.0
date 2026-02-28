import os
import shutil

def delete_annoy_files(results_dir):
    """
    Deletes files and directories within results_dir that contain 'annoy' 
    AND ('user' OR 'item') in their name (case-insensitive).
    """
    deleted_count = 0
    # Walk bottom-up so that we can delete directories without messing up the traversal
    for root, dirs, files in os.walk(results_dir, topdown=False):
        # Delete matching files
        for name in files:
            lower_name = name.lower()
            if 'annoy' in lower_name and ('user' in lower_name or 'item' in lower_name):
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Delete matching directories
        for name in dirs:
            lower_name = name.lower()
            if 'annoy' in lower_name and ('user' in lower_name or 'item' in lower_name):
                dir_path = os.path.join(root, name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")
                    
    return deleted_count

if __name__ == "__main__":
    # Get the path to the results directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    if os.path.exists(results_dir):
        print(f"Scanning '{results_dir}' for files/directories to delete...")
        count = delete_annoy_files(results_dir)
        print(f"Done. Deleted {count} items.")
    else:
        print(f"Directory '{results_dir}' does not exist.")
