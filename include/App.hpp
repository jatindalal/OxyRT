#pragma once

#include "CameraInput.hpp"
#include "Yolov10.hpp"
#include "queue.hpp"
#include "ring_buffer.hpp"
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
            auto [boxes, image] = result_or_empty.value();
            for (const auto &box : boxes) {
                float x1 = std::get<0>(box);
                float y1 = std::get<1>(box);
                float x2 = std::get<2>(box);
                float y2 = std::get<3>(box);
                float score = std::get<4>(box);
                unsigned int cls = std::get<5>(box);

                cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

                std::string label = "cls: " + std::to_string(cls) + " score: " + std::to_string(score);
                cv::putText(image, label, cv::Point(x1 + 2, y1 - 2),
                    cv::FONT_HERSHEY_DUPLEX, 2,
                    cv::Scalar(0, 0, 0), 1);
            }
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
            cv::Mat image, rgb_image;
            if (!m_camera_input.get_frame(image))
                continue;
            cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
            m_frame_buffer.push({ std::move(image), std::move(rgb_image) });
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

            auto [image, rgb_image] = frame_or_empty.value();
            auto boxes = m_model.infer(std::move(rgb_image));
            m_result_queue.push({ std::move(boxes), std::move(image) });
        }
    }

    using FrameType = std::pair<cv::Mat, cv::Mat>;
    using ResultType = std::pair<Yolov10::BoxType, cv::Mat>;
    RingBuffer<FrameType, 10> m_frame_buffer;
    ThreadSafeQueue<ResultType> m_result_queue;

    std::atomic<bool> m_shutdown { false };
    std::thread m_capture_thread;
    std::thread m_inference_thread;
    CameraInput m_camera_input;
    Yolov10 m_model;
};
