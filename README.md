# RAGBench

This project provides a **Dockerized environment** to run **RAGBench** and its evaluation scripts with minimal setup.


## 🚀 Quickstart

Build the Docker image from the project root:

```bash
docker build -t ragbench .
```


## ▶️ Run a lexical benchmark

To run a bm25 benchmark:

```bash
docker run -v "`pwd`/configs:/app/configs" -v "`pwd`/results:/app/results" -v "`pwd`/figures:/app/figures" ragbench bm25 --config configs/[your_config].yaml
```

Example with **SciDocs**:

```bash
docker run -v "`pwd`/configs:/app/configs" -v "`pwd`/results:/app/results" -v "`pwd`/figures:/app/figures" ragbench bm25 --config configs/lexical/scidocs_bm25_grid.yaml
```


## ▶️ Run a bi-encoder benchmark

To run a bi-encoder benchmark:

```bash
docker run -v "`pwd`/configs:/app/configs" -v "`pwd`/results:/app/results" -v "`pwd`/figures:/app/figures" ragbench bi_encoder --config configs/[your_config].yaml
```

Example with **SciDocs** and the **E5 model**:

```bash
docker run -v "`pwd`/configs:/app/configs" -v "`pwd`/results:/app/results" -v "`pwd`/figures:/app/figures" ragbench bi_encoder --config configs/semantic/scidocs_e5.yaml
```
