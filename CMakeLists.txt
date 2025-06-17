cmake_minimum_required(VERSION 3.10)
project(LSTMInference)

set(CMAKE_CXX_STANDARD 17)

# Adjust path based on where you extracted ONNX Runtime
set(ONNX_RUNTIME_DIR "$ENV{HOME}/onnxruntime-linux-x64-1.17.0")

include_directories(${ONNX_RUNTIME_DIR}/include)
link_directories(${ONNX_RUNTIME_DIR}/lib)

add_executable(lstm_infer src/main.cpp)
target_link_libraries(lstm_infer onnxruntime pthread)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lstm_infer OpenMP::OpenMP_CXX)
endif()
