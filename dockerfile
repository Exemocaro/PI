# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Update and upgrade
RUN apt-get update && apt-get install -y curl

# Install local model
RUN curl -L -o wavlm_rnn_model.pt 'https://drive.google.com/uc?export=download&id=1cc7V6thKP6-fxQN2nhMLq2j1qG5COgtt'

# Make port 5000 available to the world outside this container (or your port)
EXPOSE 5000

# Define environment variable (optional, useful for some servers like Flask)
ENV PYTHONUNBUFFERED=1

# Run the python server file (replace app.py with your script)
CMD ["python", "app.py"]
