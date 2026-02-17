FROM python:3.8-slim

# Add runtime libs needed by OpenCV GUI wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /main
COPY main /main
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Pre-cache CPSAM weights during build
RUN python - <<'PY'
from cellpose import models
# This will download cpsam into ~/.cellpose/models/
_ = models.CellposeModel(gpu=False, pretrained_model='cpsam')
PY

CMD ["python", "track.py"]
