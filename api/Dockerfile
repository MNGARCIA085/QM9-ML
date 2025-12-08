FROM python:3.10-slim

# Install system deps for PyTorch Geometric + RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Install torch_cluster for CPU + PyTorch 2.4.x
RUN pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI
#CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["bash"]

#docker build -f Dockerfile -t qm9:latest .
#docker run -p 8000:8000 qm9:latest


#docker run -it qm9:latest
