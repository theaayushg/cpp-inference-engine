import torch
import torch.nn as nn
import numpy as np
import onnx

# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use only the last time step
        return out

# Create model and dummy input
model = LSTMModel()
model.eval()

dummy_input = torch.randn(1, 10, 1)  # (batch, sequence_length, input_size)

# Export the model to ONNX
onnx_path = "model/lstm_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                  opset_version=12)

print(f"Exported model to: {onnx_path}")
