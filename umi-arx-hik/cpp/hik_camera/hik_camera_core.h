#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct FrameData {
    int width = 0;
    int height = 0;
    int channels = 3;
    double timestamp = 0.0;
    std::vector<uint8_t> data;
};

class HikCameraCore {
public:
    explicit HikCameraCore(const std::string& config_path);
    ~HikCameraCore();

    HikCameraCore(const HikCameraCore&) = delete;
    HikCameraCore& operator=(const HikCameraCore&) = delete;

    bool open();
    void close();

    bool start();
    void stop();

    bool is_opened() const;
    bool is_grabbing() const;

    FrameData get_frame(int timeout_ms = 1000);

    int width() const;
    int height() const;
    std::string serial_number() const;
    std::string config_path() const;

private:
    void load_config();
    void select_device();
    void create_handle();
    void open_device();
    void configure_device();
    void query_payload_size();

    void worker_loop();

    static double now_sec();
    static std::string device_serial_number(const void* dev_info);
    static std::string device_model_name(const void* dev_info);

private:
    std::string config_path_;
    std::string expected_serial_number_;

    int trigger_enable_ = 0;
    int use_ros_timestamp_ = 1;  // 保留字段，但 direct SDK 版当前不用
    int frame_rate_enable_ = 0;
    double frame_rate_ = 0.0;
    double image_scale_ = 1.0;

    int exposure_time_lower_ = 0;
    int exposure_time_upper_ = 0;
    int exposure_time_ = 0;
    int exposure_auto_mode_ = 0;
    int gain_auto_ = 0;
    float gain_ = 0.0f;
    float gamma_ = 1.0f;
    int gamma_selector_ = 0;
    int pixel_format_index_ = 0;

    void* handle_ = nullptr;
    uint32_t payload_size_ = 0;
    int width_ = 0;
    int height_ = 0;

    std::atomic<bool> opened_{false};
    std::atomic<bool> grabbing_{false};
    std::atomic<bool> worker_stop_{false};

    mutable std::mutex api_mutex_;

    std::thread worker_thread_;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FrameData latest_frame_;
    uint64_t latest_frame_seq_ = 0;
};