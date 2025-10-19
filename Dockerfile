# ===================================================================
# Base image: Ubuntu 22.04 + CUDA 12.4 + cuDNN (GPU support)
# ===================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ===================================================================
# Install Python 3.10 and build tools
# ===================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    openjdk-21-jre-headless \
    build-essential \
    ca-certificates \
    curl wget git \
 && rm -rf /var/lib/apt/lists/*

# Create symlinks so `python` and `pip` are available as commands
RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# ===================================================================
# Environment variables
# ===================================================================
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV JAVA_TOOL_OPTIONS="-Xms2g -Xmx6g"

# ===================================================================
# Working directory
# ===================================================================
WORKDIR /app

# ===================================================================
# Install Python dependencies
# ===================================================================
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# ===================================================================
# Copy project files and install ragbench CLI
# ===================================================================
COPY . /app
RUN pip install --no-cache-dir .

# ===================================================================
# Default entrypoint (shows ragbench help)
# ===================================================================
ENTRYPOINT ["ragbench"]
CMD ["--help"]
