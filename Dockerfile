##############################
# STAGE 1 — Builder
##############################
FROM python:3.10-slim AS builder

# System deps for PyG CPU + building wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install ALL dependencies here (heavy layer)
RUN pip install --no-cache-dir -r requirements.txt


##############################
# STAGE 2 — Final Runtime
##############################
FROM python:3.10-slim AS runtime

WORKDIR /app
#RUN adduser --system --no-create-home appuser
#USER appuser

# Copy only installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

##############################
# Project files (small)
##############################
COPY config/ config/
COPY tests/ tests/
COPY src/ src/
COPY scripts/ scripts/

# IMPORTANT: data is NOT copied — mount it when running
# docker run -v /mydata/qm9:/data ...

EXPOSE 8000

CMD ["bash"]




# docker build -f Dockerfile -t ml_qm9:latest .
# docker run -it ml_qm9:latest