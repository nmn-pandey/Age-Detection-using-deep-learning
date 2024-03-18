# Conda environment settings:
# Before running this cell block, create and activate a conda environment named ageDetector
# Then, run this cell block
# conda install wget 
# or
# pip install wget

# Run this cell block to download dataset (Do not edit)
import os
import wget
import zipfile

# Create a directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    print("creating directory")
    os.makedirs(data_dir)

# URL of the file to be downloaded
train_url = 'https://hr-projects-assets-prod.s3.amazonaws.com/9n6t075aj3i/a6c7b1b1d2347e59d215792fdc1a70d0/train_images.zip'
test_url = 'https://hr-projects-assets-prod.s3.amazonaws.com/9n6t075aj3i/a7172d46cd0baf561bd47c89b49f4e0f/test_images.zip'

# Directory where you want to save the file
output_dir = 'data'

# Download the file
wget.download(train_url, out=output_dir)
print("\ntraining data downloaded")
wget.download(test_url, out=output_dir)
print("\ntest data downloaded")

# Unzip the file
with zipfile.ZipFile(os.path.join(output_dir, 'train_images.zip'), 'r') as zip_ref:
    zip_ref.extractall(output_dir)
    # Check that the files were successfully unzipped
    for file in zip_ref.namelist():
        print(file)

with zipfile.ZipFile(os.path.join(output_dir, 'test_images.zip'), 'r') as zip_ref:
    zip_ref.extractall(output_dir)
    # Check that the files were successfully unzipped
    for file in zip_ref.namelist():
        print(file)

# Remove zip files
os.remove(os.path.join(output_dir, 'train_images.zip'))
os.remove(os.path.join(output_dir, 'test_images.zip'))