# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Update and upgrade the OS
RUN apt-get update && apt-get install -y curl

# Download the local model (update URL if necessary)
RUN curl -L -o whisper_base3_rnn_model.pt 'https://drive.google.com/uc?export=download&id=1cKG2sQImKSdPWv9AUZIEslRO97Rg1ozr'

# Make port 5001 available to the outside world
EXPOSE 5001

# Define environment variable (optional, useful for some servers like Flask)
ENV PYTHONUNBUFFERED=1

# Run the python server file (run.py)
CMD ["python", "run.py"]