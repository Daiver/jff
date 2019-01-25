import onnx
import sys

# Load the ONNX model
#model = onnx.load("alexnet.proto")
modelName = "torch_linear.proto"
if len(sys.argv) > 1:
    modelName = sys.argv[1]
model = onnx.load(modelName)

onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
