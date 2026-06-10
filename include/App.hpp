#pragma once

#include "CameraInput.hpp"
#include "Yolov10.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

class App {
public:
    App()
        : m_model("./models/model.onnx")
    {
        m_camera_input.open();
    }
    App(const App &) = delete;
    App &operator=(const App &) = delete;

    void run()
    {
        while (true) {
            cv::Mat image, rgb_image;
            m_camera_input.get_frame(image);
            cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
            auto boxes = m_model.infer(image);
            if (boxes.empty()) {
                continue;
            }

            for (auto &box : boxes) {
                float x1 = std::get<0>(box);
                float y1 = std::get<1>(box);
                float x2 = std::get<2>(box);
                float y2 = std::get<3>(box);
                cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("detections", image);
            cv::pollKey();
        }
    }

private:
    CameraInput m_camera_input;
    Yolov10 m_model;
};
