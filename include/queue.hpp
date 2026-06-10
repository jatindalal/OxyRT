#pragma once

#include <cstddef>
#include <deque>
#include <optional>

template <typename T, size_t SIZE = 0>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ThreadSafeQueue(const ThreadSafeQueue &) = delete;
    ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;

    void push(T&& value)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_back(std::move(value));
    }

    void push(const T& value)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_back(value);
    }
    
    std::optional<T> pop()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty()) {
            return {};
        }

        T value = std::move(m_queue.front());
        m_queue.pop_front();
        return value;
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }

private:
    std::deque<T> m_queue;
    mutable std::mutex m_mutex;
};
