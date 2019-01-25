#include <iostream>
#include <fstream>
#include <limits>

#include "onnx.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

bool parseBigMessageFromIstream(google::protobuf::Message *message, std::istream *input)
{
    google::protobuf::io::IstreamInputStream zero_copy_input(input);
    google::protobuf::io::CodedInputStream decoder(&zero_copy_input);
    const auto maxLimit = std::numeric_limits<int>::max();
    decoder.SetTotalBytesLimit(maxLimit, -1);
	const bool isParsed = message->ParseFromCodedStream(&decoder);
    const bool isConsumedEntireMessage = decoder.ConsumedEntireMessage();
    const bool isEof = input->eof();
    return isParsed && isConsumedEntireMessage && isEof;
}

void printModelInfo(const onnx::ModelProto &modelProto)
{
    std::cout 
        << "framework: "    << modelProto.producer_name() << " " 
        << "version: "      << modelProto.producer_version() << " "
        << "domain: "       << modelProto.domain() << " "
        << "model version:" << modelProto.model_version() << " "
        << modelProto.doc_string()
        << std::endl;
}

void printGraphInfo(const onnx::GraphProto &graph)
{
    std::cout << "graph name " << graph.name() << std::endl;
    std::cout << "doc str: " << graph.doc_string() << std::endl;
    std::cout << "nNodes: " << graph.node_size() << std::endl;
    std::cout << "nInitializers: " << graph.initializer_size() << std::endl;
    std::cout << "Inputs:" << std::endl;
    for(int i = 0; i < graph.input_size(); ++i){
        std::cout << "\tname: " << graph.input(i).name() << std::endl;
    }

    std::cout << "Output:" << std::endl;
    for(int i = 0; i < graph.output_size(); ++i){
        std::cout << "\tname: " << graph.output(i).name() << std::endl;
    }
}

std::string attributeToString(const onnx::AttributeProto &attr)
{
    const auto type = attr.type();

}

void printNodeInfo(const onnx::NodeProto &node)
{
    std::cout
        << "name: " << node.name()
        << "op_type: " << node.op_type()
        << "domain: " << node.domain()
        << "input: "    << node.input(0)
        << "output: "    << node.output(0)
        << std::endl;
}

int main()
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    onnx::ModelProto modelProto;
    std::fstream input("alexnet.proto", std::ios::in | std::ios::binary);
    if (!parseBigMessageFromIstream(&modelProto, &input)) {
        std::cerr << "Failed to parse model." << std::endl;
        return -1;
    }

    printModelInfo(modelProto);
    const onnx::GraphProto graph = modelProto.graph();
    printGraphInfo(graph);
    printNodeInfo(graph.node(0));

    return 0;
}

