import onnx

# Load the ONNX model
model = onnx.load("keras_linear.proto")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
