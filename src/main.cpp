#include "onnxruntime_c_api.h"
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<std::tuple<float, float, float, float, float, int>>
run_inference(Ort::Session &session, const cv::Mat &image_RGB, float min_score = 0.25f)
{
    constexpr int s_input_width = 640;
    constexpr int s_input_height = 640;
    constexpr int s_num_channels = 3;

    const float scale_x = static_cast<float>(image_RGB.cols) / static_cast<float>(s_input_width);
    const float scale_y = static_cast<float>(image_RGB.rows) / static_cast<float>(s_input_height);

    cv::Mat resized;
    cv::resize(image_RGB, resized, cv::Size(s_input_width, s_input_height));
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

    std::vector<float> input_tensor_values(1 * s_num_channels * s_input_width * s_input_height);
    // convert from HWC to CHW
    for (int y = 0; y < s_input_height; y++) {
        for (int x = 0; x < s_input_width; x++) {
            auto pixel = resized.at<cv::Vec3f>(y, x);
            input_tensor_values[0 * s_input_height * s_input_width + y * s_input_width + x] = pixel[0];
            input_tensor_values[1 * s_input_height * s_input_width + y * s_input_width + x] = pixel[1];
            input_tensor_values[2 * s_input_height * s_input_width + y * s_input_width + x] = pixel[2];
        }
    }
    const std::array<int64_t, 4> input_shape({ 1, s_num_channels, s_input_height, s_input_width });

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
        input_tensor_values.size(), input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

    const char *input_names[] = { input_name_ptr.get() };
    const char *output_names[] = { output_name_ptr.get() };

    auto output_tensors = session.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, 1);
    if (output_tensors.empty())
        return { };

    auto &output_tensor = output_tensors[0];
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    float *output_data = output_tensor.GetTensorMutableData<float>();

    const int detections = static_cast<int>(shape[1]);
    std::vector<std::tuple<float, float, float, float, float, int>> boxes;
    for (int i = 0; i < detections; i++) {
        const float *det = output_data + i * 6;

        int cls = static_cast<int>(det[5]);

        float score = det[4];
        if (score < min_score) {
            continue;
        }

        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];

        float orig_x1 = x1 * scale_x;
        float orig_y1 = y1 * scale_y;
        float orig_x2 = x2 * scale_x;
        float orig_y2 = y2 * scale_y;

        boxes.emplace_back(orig_x1, orig_y1, orig_x2, orig_y2, score, cls);
    }
    return boxes;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image path>\n";
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (!image.data) {
        throw std::runtime_error("No image data");
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::Session session(env, "./models/model.onnx", { });

    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    auto boxes = run_inference(session, image);
    if (boxes.empty()) {
        std::cerr << "No boxes found for the image!";
    }

    for (auto &box : boxes) {
        float x1 = std::get<0>(box);
        float y1 = std::get<1>(box);
        float x2 = std::get<2>(box);
        float y2 = std::get<3>(box);
        std::cout << "cls=" << std::get<5>(box) << " score=" << std::get<4>(box)
                  << " box=(" << std::get<0>(box) << ", " << std::get<1>(box) << ", "
                  << std::get<2>(box) << ", " << std::get<3>(box) << ")\n";
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("detections", image);
    cv::waitKey(0);

    return 0;
}
