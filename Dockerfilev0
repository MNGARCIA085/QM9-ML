FROM python:3.10-slim

# ===== System dependencies for PyTorch Geometric + RDKit =====
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install base Python deps
RUN pip install -r requirements.txt


COPY config/ config
COPY tests/ tests
COPY src/ src/
COPY scripts/ scripts/


EXPOSE 8000

CMD ["bash"]



# docker build -f Dockerfile -t ml_qm9:latest .
# docker run -it ml_qm9:latest



# Install PyTorch Geometric ops (CPU wheels for PyTorch 2.4.x)
#RUN pip install \
#    torch_cluster \
#    torch_scatter==2.1.2+pt28cpu \
#    torch_sparse==0.6.18+pt28cpu \
#    torch_spline_conv==1.2.2+pt28cpu \
#    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html