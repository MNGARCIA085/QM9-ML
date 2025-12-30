##############################
# STAGE 1 — Builder
##############################
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


##############################
# STAGE 2 — Final Runtime
##############################
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed dependencies
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

##############################
# Project files
##############################
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY config/ config/

# INSTALL YOUR PACKAGE
RUN pip install -e . # uses the toml to know is python package

EXPOSE 8000
CMD ["bash"]



# docker build -f Dockerfile -t ml_qm9:latest .
# docker run -it ml_qm9:latest