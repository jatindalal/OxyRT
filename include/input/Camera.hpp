#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

class CameraInput {
public:
    CameraInput() = default;
    CameraInput(const CameraInput &) = delete;
    CameraInput &operator=(const CameraInput &) = delete;

    void open(int index = 0)
    {
        if (!m_capture_device.open(index)) {
            throw std::runtime_error("Failed to open Webcam!");
        }
        m_capture_device.set(cv::CAP_PROP_BUFFERSIZE, 1.0);
        m_opened = true;
    }

    bool get_frame(cv::Mat &output)
    {
        if (!m_opened) {
            throw std::runtime_error("get_frame called without opening a device!");
        }
        return m_capture_device.read(output);
    }

private:
    cv::VideoCapture m_capture_device;
    bool m_opened = false;
};
