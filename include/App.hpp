#pragma once

#include "Yolov10.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

class App {
public:
    App() = default;
    App(const App &) = delete;
    App &operator=(const App &) = delete;

    void run(std::string image_path)
    {
        Yolov10 yolo_model("./models/model.onnx");

        cv::Mat image;
        image = cv::imread(image_path, cv::IMREAD_COLOR);

        if (!image.data) {
            throw std::runtime_error("No image data");
        }

        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        auto boxes = yolo_model.infer(image);
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
    }

private:
};
