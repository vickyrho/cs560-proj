# Use a Python base image
FROM python:3.10-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Add Z3 to the PATH
ENV PATH="/opt/z3/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebook and the data file into the container
COPY used_cars_notebook.ipynb /app/notebook.ipynb
COPY used_cars_india.csv /app/used_cars_india.csv

# (Optional) Install Jupyter if you want to run the notebook interactively
RUN pip install jupyter

# Specify the command to execute the notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

