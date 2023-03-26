from huggingface_hub import login, Repository
import os


login()
repo = Repository("MortenTabaka/LandCover-Aerial-Imagery-for-semantic-segmentation")

# Directory containing images
abs_path = os.path.abspath("")
images_dir = abs_path + "/data/processed"

# Create a directory in your repository to store the images
repo.create_directory("processed")


# Recursive function to upload files and preserve directory structure
def upload_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Get the relative path of the file within the "processed" directory
            rel_path = os.path.relpath(file_path, images_dir)
            # Construct the destination path in the repository
            dest_path = os.path.join("processed", rel_path)
            with open(file_path, "rb") as f:
                repo.upload_file(f, dest_path)
        elif os.path.isdir(file_path):
            # Recursively upload files in subdirectories
            upload_files(file_path)


# Upload files starting from the "processed" directory
upload_files(images_dir)
