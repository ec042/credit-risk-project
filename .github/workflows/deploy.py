import shutil
import os

# Define the source and destination paths
source_path = 'credit.pkl'
destination_dir = 'deployment_dir'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Copy the model file to the destination directory
shutil.copy(source_path, destination_dir)

print(f'Model deployed to {destination_dir}')
