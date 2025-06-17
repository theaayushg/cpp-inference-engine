```markdown
# High-Performance Neural Network Inference Engine (C++ / ONNX / OpenMP)

This project implements a low-latency neural network inference engine in C++ using ONNX Runtime.  
It loads an LSTM model trained and exported in PyTorch, performs inference using ONNX Runtime’s C++ API, and leverages OpenMP for multithreaded performance under simulated streaming workloads.

---

## 🔧 Features

- ✅ C++-based inference with [ONNX Runtime](https://onnxruntime.ai/)
- ✅ PyTorch → ONNX conversion for LSTM time-series models
- ✅ OpenMP multithreading to improve throughput
- ✅ Achieves sub-2ms average latency on standard laptop CPU
- ✅ Fully portable and dependency-light (no CUDA or training code needed at runtime)

---

## 🛠️ Build Instructions

### Prerequisites
- Ubuntu (WSL or native)
- CMake ≥ 3.10
- GCC ≥ 9
- [ONNX Runtime C++ SDK](https://github.com/microsoft/onnxruntime/releases) v1.17.0 (precompiled)

### Steps
1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cpp-inference-engine.git
   cd cpp-inference-engine
   ```

2. Download ONNX Runtime SDK:
   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
   tar -xvzf onnxruntime-linux-x64-1.17.0.tgz
   ```

3. Build:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. Run:
   ```bash
   ./lstm_infer
   ```

---

## 📦 Project Structure

```
cpp-inference-engine/
├── model/               # Contains exported ONNX model
│   └── lstm_model.onnx
├── src/
│   └── main.cpp         # C++ inference logic
├── export_lstm.py       # PyTorch LSTM model + ONNX export
├── CMakeLists.txt       # Build config
```

---

## 📊 Sample Output

```
Model loaded successfully!
=== OpenMP Inference Benchmark ===
Thread 0: Output = -0.0034, Time = 1315 µs
...
Average inference time: 1419 µs
```

---

## 🔬 Technologies Used

- C++17
- ONNX Runtime C++ API
- PyTorch (for model export only)
- OpenMP (parallel inference)
- CMake

---

## 🧠 Use Cases

- Real-time financial forecasting
- Quant research infrastructure
- Latency-sensitive edge inference
```
