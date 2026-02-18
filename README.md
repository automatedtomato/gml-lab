# GML-Lab (Green-ML Optimization Lab)

**GML-Lab** is an experimental playground (laboratory) designed to dissect, measure, and optimize **`mmpretrain`** models.
Unlike a black-box conversion tool, this project serves as a workbench for bridging high-level model abstractions (PyTorch/FX) and low-level hardware implementations (Custom CUDA Kernels) on consumer hardware (NVIDIA RTX 3060 Ti).

The primary goal is to establish a feedback loop of **"Measure (Profile) -> Analyze (SNR/Sensitivity) -> Optimize (Quantize/Kernel)"** to achieve Green AI principles.

## Key Features

* **FX Graph Quantization**: Transparent PTQ pipeline using `torch.fx` with graph visualization.
* **Micro-Benchmarking**: Custom profiler for tracking layer-wise latency (CPU/GPU) to identify real bottlenecks.
* **Sensitivity Analysis**: Statistical analysis (SNR, Cosine Similarity) to visualize quantization damage per layer.
* **High-Throughput I/O**: Optimized data loading via LMDB to prevent GPU starvation during profiling.

## Roadmap (Experiments in Queue)

This lab is currently in **Sprint 2** phase. The following features are under active development:

* [x] **Baseline Measurement**: FP32 vs. INT8 accuracy & latency profiling.
* [ ] **Custom CUDA Kernels**: Implementing fused kernels for identified bottlenecks (e.g., LayerNorm, Attention).
* [ ] **Mixed Precision**: Surgical FP16 fallback for sensitive layers (AutoMP).
* [ ] **Transformer Support**: Graph tracing patches for ViT/Swin Transformers.

## Limitations & Workarounds

**`torch.fx` Symbol Tracing**:
This project relies heavily on `torch.fx.symbolic_trace`. However, many open-source models (including `mmpretrain`) contain dynamic control flows or non-standard Python features that cause tracing to fail.

* **Behavior**: If you encounter a `TraceError`, the model likely requires modification.
* **Solution**: We apply **Monkey Patching** to replace non-traceable components with FX-friendly equivalents. Check `src/gml_lab/modeling/` for existing patches. If you bring a new architecture, you may need to write a custom patch.

---

## 1. Installation

The environment is fully containerized to ensure reproducibility.

### Prerequisites

* NVIDIA Driver (535+)
* Docker & NVIDIA Container Toolkit

### Setup

1. **Build the Docker image**:
```bash
docker build -t gml-lab -f docker/Dockerfile .
```


2. **Start the container**:
```bash
# Mount current directory. --privileged is required for NVML/Power profiling.
docker run --gpus all -it --ipc=host --privileged \
  -v $(pwd):/gml-lab \
  gml-lab bash
```


3. **Install dependencies (inside container)**:
```bash
pip install -e .
```



## 2. Data Preparation

GML-Lab uses **LMDB** for efficient random access. You must convert your dataset (ImageNet-style structure) before running benchmarks.

### Directory Structure

```text
data/raw/imagenet/val/
├── n01440764/
│   ├── ILSVRC2012_val_00000293.JPEG
│   └── ...
```

### Conversion Command

```bash
python scripts/data_conversion.py \
  --data-root data/raw/imagenet/val \
  --out-path data/imagenet/val_lmdb
```

> **Note**: Update `configs/imagenet_lmdb.py` if you use a different path.

## 3. Usage

The main entry point is `examples/gml_example.py`.

### Basic Evaluation

Compare **FP32 (Baseline)** and **Int8 (QDQ)** accuracy.

```bash
python -m examples.gml_example float qdq \
  --arch resnet18_8xb32_in1k \
  --batch-size 64
```

### Profiling (Latency)

Measure layer-wise execution time on the GPU. This helps decide which layers to optimize with custom kernels.

```bash
python -m examples.gml_example --enable-profile
```

**Output**: `examples/results/{arch}/qdq_prof.json`

### Sensitivity Analysis

Generate a report comparing FP32 vs. Int8 activations to detect accuracy degradation sources.

```bash
python -m examples.gml_example \
  --lbl-dump-dir examples/results/analysis/

```

### Visualization

Dump the computational graph (Dot/PDF) for debugging `torch.fx` transformations.

```bash
python -m examples.gml_example \
  --graph-dump-dir examples/results/graphs/

```

## Directory Structure

```text
gml-lab/
├── configs/            # MMPretrain & Data configurations
├── csrc/               # Custom CUDA kernels (C++/CUDA) Source
├── examples/           # Experiment runners
├── src/
│   └── gml_lab/
│       ├── modeling/   # Model loading & FX patching logic
│       ├── quantizer/  # FX Graph Quantization pipeline
│       └── ...
├── tests/              # Unit tests
└── tools/              # Profiler & Analysis tools

```
