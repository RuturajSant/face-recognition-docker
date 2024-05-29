FROM python:3.9-slim

WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

COPY ./spoof /app/spoof

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask app
CMD ["flask", "run"]