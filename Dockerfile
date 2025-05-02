# Use an official Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirement files and install
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install OS-level dependencies if needed
COPY packages.txt ./
RUN xargs -r -a packages.txt apt-get update && xargs -r -a packages.txt apt-get install -y && apt-get clean

# Copy the rest of the app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
