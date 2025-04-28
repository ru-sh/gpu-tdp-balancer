FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# pynvml is the key dependency for NVML interaction
RUN pip3 install pynvml

# Copy the application code into the container
COPY src/gpu_tdp_balancer /app/gpu_tdp_balancer
COPY src/main.py /app/main.py

# Set the working directory
WORKDIR /app

# Set the entrypoint to run the main script
# Running as root inside the container is necessary because the application
# requires root privileges on the host to set power limits via NVML.
# Be aware of the security implications of running containers as root.
ENTRYPOINT ["python3", "main.py"]