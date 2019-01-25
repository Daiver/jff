main: onnx.pb.o
	g++ -o main main.cpp onnx.pb.o -L/usr/local/lib -lprotobuf
onnx.pb.o:
	g++ -c -o onnx.pb.o onnx.pb.cc 
