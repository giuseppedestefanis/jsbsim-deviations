# Replication image for "Detecting behavioural changes in flight dynamics model updates" (JSBSim study).
# Build: docker build -t jsbsim-deviations .   Run: docker run --rm -it jsbsim-deviations bash
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY . /work
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir jsbsim==1.3.1 pandas==3.0.5 numpy==2.5.3 scikit-learn==1.9.0 scipy==1.18.1 matplotlib==3.11.1 pyarrow==25.0.1
ENV PATH="/work/.venv/bin:$PATH" OMP_NUM_THREADS=1
CMD ["bash"]
