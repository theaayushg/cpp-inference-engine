#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <omp.h>  // OpenMP

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lstm_infer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);  // Let OpenMP handle threading
    Ort::Session session(env, "../model/lstm_model.onnx", session_options);
    std::cout << "Model loaded successfully!" << std::endl;

    // Input shape: [1, 10, 1]
    std::vector<int64_t> input_shape = {1, 10, 1};
    std::vector<float> input_tensor_values(1 * 10 * 1, 0.5f);  // dummy data
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    int num_threads = 8;
    std::vector<float> outputs(num_threads);
    std::vector<long long> durations(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto end = std::chrono::high_resolution_clock::now();

        float* float_array = output_tensors.front().GetTensorMutableData<float>();
        outputs[i] = float_array[0];
        durations[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    // Summary
    std::cout << "=== OpenMP Inference Benchmark ===" << std::endl;
    for (int i = 0; i < num_threads; ++i) {
        std::cout << "Thread " << i << ": Output = " << outputs[i]
                  << ", Time = " << durations[i] << " µs" << std::endl;
    }

    long long total = 0;
    for (auto d : durations) total += d;
    std::cout << "Average inference time: " << total / num_threads << " µs" << std::endl;

    return 0;
}
