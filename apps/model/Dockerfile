FROM python:3.8-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir numpy torch e3nn matplotlib
RUN pip install --no-cache-dir torch_scatter torch_cluster torch_sparse torch_spline_conv
RUN pip install --no-cache-dir torch_geometric
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
