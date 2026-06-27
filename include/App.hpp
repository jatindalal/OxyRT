#pragma once

#include "input/Camera.hpp"
#include "inference/Yolov10.hpp"
#include "core/queue.hpp"
#include "core/ring_buffer.hpp"
#include <atomic>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

class App {
public:
    App()
        : m_model("./models/model.onnx")
        , m_result_queue(5)
    {
    }
    App(const App &) = delete;
    App &operator=(const App &) = delete;

    void run()
    {
        m_camera_input.open();
        start_workers();
        run_ui();
        close_workers();
    }

private:
    void start_workers()
    {
        m_capture_thread = std::thread([this]() { capture_loop(); });
        m_inference_thread = std::thread([this]() { inference_loop(); });
    }

    void run_ui()
    {
        while (!m_shutdown.load()) {
            auto result_or_empty = m_result_queue.pop();
            if (!result_or_empty.has_value()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto result = std::move(result_or_empty.value());
            const auto& boxes = result.m_detections;
            const auto& image = result.m_image;

            for (const auto &box : boxes) {
                cv::rectangle(image, cv::Point(box.m_x1, box.m_y1), cv::Point(box.m_x2, box.m_y2), cv::Scalar(0, 255, 0), 2);
                std::string label = "cls: " + std::to_string(box.m_cls) + " score: " + std::to_string(box.m_confidence);
                cv::putText(image, label, cv::Point(box.m_x1 + 2, box.m_y1 - 2),
                    cv::FONT_HERSHEY_DUPLEX, 2,
                    cv::Scalar(0, 0, 0), 1);
            }
            cv::namedWindow("detections", cv::WINDOW_NORMAL);
            cv::imshow("detections", image);
            if (cv::pollKey() >= 0) {
                m_shutdown = true;
            };
        }
    }

    void close_workers()
    {
        m_shutdown.store(true);
        m_capture_thread.join();
        m_inference_thread.join();
    }

    void capture_loop()
    {
        while (!m_shutdown.load()) {
            cv::Mat image;
            if (!m_camera_input.get_frame(image))
                continue;
            m_frame_buffer.push(std::move(image));
        }
    }

    void inference_loop()
    {
        while (!m_shutdown.load()) {
            auto frame_or_empty = m_frame_buffer.pop();
            if (!frame_or_empty.has_value()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            auto image = frame_or_empty.value();
            auto result = m_model.infer(std::move(image));
            m_result_queue.push(std::move(result));
        }
    }

    using FrameType = cv::Mat;
    using ResultType = Yolov10::DetectionResult;
    RingBuffer<FrameType, 10> m_frame_buffer;
    ThreadSafeQueue<ResultType> m_result_queue;

    std::atomic<bool> m_shutdown { false };
    std::thread m_capture_thread;
    std::thread m_inference_thread;
    CameraInput m_camera_input;
    Yolov10 m_model;
};
