# Start from a PyTorch base image with Python 3.8 and CUDA 12.0 support
# PyTorch's official Docker images usually come with both Python and CUDA installed
FROM pytorch/pytorch:2.1.1-cuda12.0-cudnn8-runtime

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
# Note: Since PyTorch is already installed, make sure it's not in requirements.txt or use '--no-deps' to avoid re-installing it
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run your application
# Replace 'your_script.py' with the script you want to run, e.g., 'train.py'
CMD ["python", "your_script.py"]
