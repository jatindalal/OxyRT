#pragma once
#include <cstddef>
#include <cstdint>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#ifdef USE_COREML
#include <coreml_provider_factory.h>
#endif

class Yolov10 {
public:
    using BoxType = std::vector<std::tuple<float, float, float, float, float, unsigned int>>;
    Yolov10(std::string model_path)
        : m_model_path(model_path)
        , m_ort_env(ORT_LOGGING_LEVEL_WARNING, "yolov10")
        , m_session(create_session())
    {
    }

    BoxType infer(const cv::Mat &image_RGB, float min_score = 0.25f)
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
        auto input_name_ptr = m_session.GetInputNameAllocated(0, allocator);
        auto output_name_ptr = m_session.GetOutputNameAllocated(0, allocator);

        const char *input_names[] = { input_name_ptr.get() };
        const char *output_names[] = { output_name_ptr.get() };

        auto output_tensors = m_session.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, 1);
        if (output_tensors.empty())
            return { };

        auto &output_tensor = output_tensors[0];
        auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        float *output_data = output_tensor.GetTensorMutableData<float>();

        const int detections = static_cast<int>(shape[1]);
        BoxType boxes;
        for (int i = 0; i < detections; i++) {
            const float *det = output_data + i * 6;

            unsigned int cls = static_cast<unsigned int>(det[5]);

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

private:
    Ort::Session create_session()
    {
        Ort::SessionOptions session_options;
        auto providers = Ort::GetAvailableProviders();
#ifdef USE_COREML
        if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") != providers.end()) {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0));
            std::cout << "Using CoreML\n";
        }
#endif
#ifdef USE_CUDA
        if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
            std::cout << "Using CUDA\n";
        }
#endif
        return Ort::Session(m_ort_env, m_model_path.c_str(), session_options);
    }
    std::string m_model_path;
    Ort::Env m_ort_env;
    Ort::Session m_session;
};
