import onnx
import onnxruntime

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# onnx test
alexnet_session = onnxruntime.InferenceSession("alexnet.onnx")
print(alexnet_session)

