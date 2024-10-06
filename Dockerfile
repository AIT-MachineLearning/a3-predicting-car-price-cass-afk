# Use an official Python runtime as a parent image
FROM python:3.11.3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7070 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME=DashApp

# Use JSON format for CMD to avoid unintended behavior
CMD ["python", "app.py"]
