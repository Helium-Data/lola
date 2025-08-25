# Use an official Python image as a base
FROM python:3.12-slim

# revisit using alpine base image to reduce image size
# FROM python:3.13-alpine
# FROM python:3.12-alpine
# Install dependencies
# RUN apk add --no-cache g++ gcc musl-dev libffi-dev openssl-dev
RUN apt-get update && apt-get install build-essential -y

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

RUN pip install --upgrade pip

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port used by the FastAPI app
EXPOSE 3000

# Run the command to start the app when the container launches
CMD ["uvicorn", "app.client:fast_app", "--host", "0.0.0.0", "--port", "3000"]
