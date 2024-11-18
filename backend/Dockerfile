# Use the official FastAPI uvicorn image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
