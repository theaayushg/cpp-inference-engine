# High-Performance Neural Network Inference Engine (C++ / ONNX / OpenMP)

This project implements a low-latency inference engine in C++ using ONNX Runtime. It loads an LSTM model trained in PyTorch, exported to ONNX, and runs multithreaded inference using OpenMP.

## Features
- ✅ C++ inference with ONNX Runtime
- ✅ PyTorch-to-ONNX LSTM export
- ✅ OpenMP multithreading for high-throughput
- ✅ Sub-2ms latency on standard CPU

## Setup
1. Clone this repo
2. Download ONNX Runtime C++ SDK: [Releases](https://github.com/microsoft/onnxruntime/releases)
3. Build with CMake:
