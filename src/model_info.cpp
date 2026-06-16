#include <onnxruntime_cxx_api.h>

#include <iostream>

static void print_shape(const std::vector<int64_t> &shape)
{
    std::cout << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];

        if (i + 1 < shape.size())
            std::cout << ", ";
    }

    std::cout << "]";
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions opts;
    Ort::Session session(env, argv[1], opts);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t input_count = session.GetInputCount();
    size_t output_count = session.GetOutputCount();

    std::cout << "Input Count: " << input_count << std::endl;
    std::cout << "Output Count: " << output_count << std::endl;

    for (size_t i = 0; i < input_count; i++) {
        auto name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Input: " << i << std::endl;
        std::cout << "\tName: " << name.get() << "\n";
        std::cout << "\tRank: " << shape.size() << "\n";
        std::cout << "\tShape: ";
        print_shape(shape);
        std::cout << std::endl;
        std::cout << "\tElement Type: " << tensor_info.GetElementType() << std::endl;
    }

    for (size_t i = 0; i < output_count; i++) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Output: " << i << std::endl;
        std::cout << "\tName: " << name.get() << "\n";
        std::cout << "\tRank: " << shape.size() << "\n";
        std::cout << "\tShape: ";
        print_shape(shape);
        std::cout << std::endl;
        std::cout << "\tElement Type: " << tensor_info.GetElementType() << std::endl;
    }

    return 0;
}
