#include "hik_camera_core.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

namespace {

enum PixelFormat : unsigned int {
    RGB8 = 0x02180014,
    BayerRG8 = 0x01080009,
    BayerRG12Packed = 0x010C002B,
    BayerGB12Packed = 0x010C002C,
    BayerGB8 = 0x0108000A
};

static const std::vector<unsigned int> PIXEL_FORMATS = {
    RGB8,
    BayerRG8,
    BayerRG12Packed,
    BayerGB12Packed,
    BayerGB8
};

static const char* ExposureAutoStr[3] = {"Off", "Once", "Continues"};
static const char* GammaSelectorStr[3] = {"User", "sRGB", "Off"};
static const char* GainAutoStr[3] = {"Off", "Once", "Continues"};

static std::string hex_ret(int ret) {
    std::ostringstream oss;
    oss << "0x" << std::hex << ret;
    return oss.str();
}

static void check_mv(int ret, const std::string& msg) {
    if (ret != MV_OK) {
        throw std::runtime_error(msg + " failed, ret=" + hex_ret(ret));
    }
}

static void print_warn(const std::string& msg) {
    std::cerr << "[HikCamera] Warning: " << msg << std::endl;
}

static void print_info(const std::string& msg) {
    std::cout << "[HikCamera] " << msg << std::endl;
}

}  // namespace

HikCameraCore::HikCameraCore(const std::string& config_path)
    : config_path_(config_path) {
    load_config();
}

HikCameraCore::~HikCameraCore() {
    try {
        stop();
        close();
    } catch (...) {
    }
}

double HikCameraCore::now_sec() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string HikCameraCore::config_path() const {
    return config_path_;
}

std::string HikCameraCore::serial_number() const {
    return expected_serial_number_;
}

int HikCameraCore::width() const {
    return width_;
}

int HikCameraCore::height() const {
    return height_;
}

bool HikCameraCore::is_opened() const {
    return opened_.load();
}

bool HikCameraCore::is_grabbing() const {
    return grabbing_.load();
}

void HikCameraCore::load_config() {
    cv::FileStorage fs(config_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open config file: " + config_path_);
    }

    expected_serial_number_ = (std::string)fs["SerialNumber"];
    trigger_enable_ = fs["TriggerEnable"].empty() ? 0 : (int)fs["TriggerEnable"];
    use_ros_timestamp_ = fs["UseROSTimestamp"].empty() ? 1 : (int)fs["UseROSTimestamp"];

    frame_rate_enable_ = fs["FrameRateEnable"].empty() ? 0 : (int)fs["FrameRateEnable"];
    frame_rate_ = fs["FrameRate"].empty() ? 0.0 : (double)fs["FrameRate"];

    image_scale_ = fs["image_scale"].empty() ? 1.0 : (double)fs["image_scale"];
    if (image_scale_ < 0.1) image_scale_ = 1.0;

    exposure_time_lower_ = fs["AutoExposureTimeLower"].empty() ? 0 : (int)fs["AutoExposureTimeLower"];
    exposure_time_upper_ = fs["AutoExposureTimeUpper"].empty() ? 0 : (int)fs["AutoExposureTimeUpper"];
    exposure_time_ = fs["ExposureTime"].empty() ? 0 : (int)fs["ExposureTime"];
    exposure_auto_mode_ = fs["ExposureAutoMode"].empty() ? 0 : (int)fs["ExposureAutoMode"];
    gain_auto_ = fs["GainAuto"].empty() ? 0 : (int)fs["GainAuto"];
    gain_ = fs["Gain"].empty() ? 0.0f : (float)(double)fs["Gain"];
    gamma_ = fs["Gamma"].empty() ? 1.0f : (float)(double)fs["Gamma"];
    gamma_selector_ = fs["GammaSelector"].empty() ? 0 : (int)fs["GammaSelector"];
    pixel_format_index_ = fs["PixelFormat"].empty() ? 0 : (int)fs["PixelFormat"];

    if (pixel_format_index_ < 0 || pixel_format_index_ >= (int)PIXEL_FORMATS.size()) {
        throw std::runtime_error(
            "Invalid PixelFormat index in YAML: " + std::to_string(pixel_format_index_));
    }
}

std::string HikCameraCore::device_serial_number(const void* dev_info_void) {
    const auto* dev_info = reinterpret_cast<const MV_CC_DEVICE_INFO*>(dev_info_void);
    if (dev_info->nTLayerType == MV_USB_DEVICE) {
        return std::string(
            reinterpret_cast<const char*>(dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber));
    } else if (dev_info->nTLayerType == MV_GIGE_DEVICE) {
        return std::string(
            reinterpret_cast<const char*>(dev_info->SpecialInfo.stGigEInfo.chSerialNumber));
    }
    return "";
}

std::string HikCameraCore::device_model_name(const void* dev_info_void) {
    const auto* dev_info = reinterpret_cast<const MV_CC_DEVICE_INFO*>(dev_info_void);
    if (dev_info->nTLayerType == MV_USB_DEVICE) {
        return std::string(
            reinterpret_cast<const char*>(dev_info->SpecialInfo.stUsb3VInfo.chModelName));
    } else if (dev_info->nTLayerType == MV_GIGE_DEVICE) {
        return std::string(
            reinterpret_cast<const char*>(dev_info->SpecialInfo.stGigEInfo.chModelName));
    }
    return "Unknown";
}

void HikCameraCore::select_device() {
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    std::memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    check_mv(nRet, "MV_CC_EnumDevices");

    if (stDeviceList.nDeviceNum == 0) {
        throw std::runtime_error("Find No Devices!");
    }

    std::cout << "[HikCamera] Found " << stDeviceList.nDeviceNum << " device(s)" << std::endl;
    for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++) {
        MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
        if (pDeviceInfo == nullptr) continue;
        std::cout << "  [device " << i << "] model="
                  << device_model_name(pDeviceInfo)
                  << " serial=" << device_serial_number(pDeviceInfo)
                  << std::endl;
    }

    bool find_expect_camera = false;
    unsigned int nIndex = 0;

    if (stDeviceList.nDeviceNum > 1) {
        if (expected_serial_number_.empty()) {
            throw std::runtime_error("Expected serial number is empty!");
        }

        for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++) {
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (pDeviceInfo == nullptr) continue;

            std::string serial_number = device_serial_number(pDeviceInfo);
            if (serial_number.empty()) continue;

            if (expected_serial_number_ == serial_number) {
                find_expect_camera = true;
                nIndex = i;
                break;
            }
        }

        if (!find_expect_camera) {
            throw std::runtime_error(
                "Can not find the camera with serial number " + expected_serial_number_);
        }
    } else {
        nIndex = 0;
    }

    if (handle_ != nullptr) {
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }

    nRet = MV_CC_CreateHandle(&handle_, stDeviceList.pDeviceInfo[nIndex]);
    check_mv(nRet, "MV_CC_CreateHandle");

    std::cout << "[HikCamera] Selected device index " << nIndex
              << ", serial=" << device_serial_number(stDeviceList.pDeviceInfo[nIndex])
              << std::endl;
}

void HikCameraCore::create_handle() {
    // reserved
}

void HikCameraCore::open_device() {
    int nRet = MV_CC_OpenDevice(handle_);
    check_mv(nRet, "MV_CC_OpenDevice");
}

void HikCameraCore::configure_device() {
    int nRet = MV_OK;

    int frame_rate_enable = frame_rate_enable_;
    double frame_rate = frame_rate_;

    nRet = MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", frame_rate_enable != 0);
    if (MV_OK != nRet) {
        throw std::runtime_error("set AcquisitionFrameRateEnable fail! nRet [" + hex_ret(nRet) + "]");
    }

    if (frame_rate_enable != 0) {
        nRet = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", (float)frame_rate);
        if (MV_OK != nRet) {
            print_warn("set AcquisitionFrameRate fail! nRet [" + hex_ret(nRet) + "]");
        } else {
            print_info("Set AcquisitionFrameRate: " + std::to_string(frame_rate));
        }
    }

    nRet = MV_CC_SetEnumValue(handle_, "PixelFormat", PIXEL_FORMATS[pixel_format_index_]);
    if (nRet != MV_OK) {
        throw std::runtime_error("Pixel setting can't work. nRet [" + hex_ret(nRet) + "]");
    }

    // ---- setParams(handle, params_file) equivalent
    nRet = MV_CC_SetExposureAutoMode(handle_, exposure_auto_mode_);
    {
        std::string msg = "Set ExposureAutoMode: " +
                          std::string(ExposureAutoStr[
                              (exposure_auto_mode_ >= 0 && exposure_auto_mode_ <= 2)
                                  ? exposure_auto_mode_
                                  : 0]);
        if (MV_OK == nRet) {
            print_info(msg);
        } else {
            if (exposure_auto_mode_ == 2) {
                print_warn("Fail to set Exposure Auto Mode to Continues");
            } else {
                print_info(msg);
            }
        }
    }

    if (exposure_auto_mode_ == 2) {
        nRet = MV_CC_SetAutoExposureTimeLower(handle_, exposure_time_lower_);
        if (MV_OK == nRet) {
            print_info("Set Exposure Time Lower: " + std::to_string(exposure_time_lower_) + "us");
        } else {
            print_warn("Fail to set Exposure Time Lower");
        }

        nRet = MV_CC_SetAutoExposureTimeUpper(handle_, exposure_time_upper_);
        if (MV_OK == nRet) {
            print_info("Set Exposure Time Upper: " + std::to_string(exposure_time_upper_) + "us");
        } else {
            print_warn("Fail to set Exposure Time Upper");
        }
    }

    if (exposure_auto_mode_ == 0) {
        nRet = MV_CC_SetExposureTime(handle_, exposure_time_);
        if (MV_OK == nRet) {
            print_info("Set Exposure Time: " + std::to_string(exposure_time_) + "us");
        } else {
            print_warn("Fail to set Exposure Time");
        }
    }

    nRet = MV_CC_SetEnumValue(handle_, "GainAuto", gain_auto_);
    if (MV_OK == nRet) {
        print_info("Set Gain Auto: " +
                   std::string(GainAutoStr[(gain_auto_ >= 0 && gain_auto_ <= 2) ? gain_auto_ : 0]));
    } else {
        print_warn("Fail to set Gain auto mode");
    }

    if (gain_auto_ == 0) {
        nRet = MV_CC_SetGain(handle_, gain_);
        if (MV_OK == nRet) {
            print_info("Set Gain: " + std::to_string(gain_));
        } else {
            print_warn("Fail to set Gain");
        }
    }

    nRet = MV_CC_SetGammaSelector(handle_, gamma_selector_);
    if (MV_OK == nRet) {
        print_info("Set GammaSelector: " +
                   std::string(GammaSelectorStr[
                       (gamma_selector_ >= 0 && gamma_selector_ <= 2)
                           ? gamma_selector_
                           : 0]));
    } else {
        print_warn("Fail to set GammaSelector");
    }

    nRet = MV_CC_SetGamma(handle_, gamma_);
    if (MV_OK == nRet) {
        print_info("Set Gamma: " + std::to_string(gamma_));
    } else {
        print_warn("Fail to set Gamma");
    }

    nRet = MV_CC_SetEnumValue(handle_, "TriggerMode", trigger_enable_);
    if (MV_OK != nRet) {
        throw std::runtime_error("MV_CC_SetTriggerMode fail! nRet [" + hex_ret(nRet) + "]");
    }

    nRet = MV_CC_SetEnumValue(handle_, "TriggerSource", MV_TRIGGER_SOURCE_LINE2);
    if (MV_OK != nRet) {
        throw std::runtime_error("MV_CC_SetTriggerSource fail! nRet [" + hex_ret(nRet) + "]");
    }

    print_info("Finish all params set! Ready to start grabbing...");
}

void HikCameraCore::query_payload_size() {
    MVCC_INTVALUE stParam;
    std::memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    int nRet = MV_CC_GetIntValue(handle_, "PayloadSize", &stParam);
    if (MV_OK != nRet) {
        throw std::runtime_error("Get PayloadSize fail! nRet [" + hex_ret(nRet) + "]");
    }

    payload_size_ = stParam.nCurValue;
    print_info("PayloadSize = " + std::to_string(payload_size_));
}

bool HikCameraCore::open() {
    std::lock_guard<std::mutex> lock(api_mutex_);
    if (opened_) return true;

    select_device();
    open_device();
    configure_device();

    opened_ = true;
    return true;
}

bool HikCameraCore::start() {
    std::lock_guard<std::mutex> lock(api_mutex_);
    if (!opened_) {
        throw std::runtime_error("Camera must be opened before start()");
    }
    if (grabbing_) return true;

    if (payload_size_ == 0) {
        query_payload_size();
    }

    int nRet = MV_CC_StartGrabbing(handle_);
    if (MV_OK != nRet) {
        throw std::runtime_error("Start Grabbing fail. nRet [" + hex_ret(nRet) + "]");
    }

    worker_stop_.store(false);
    grabbing_.store(true);

    // reset latest frame state
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = FrameData();
        latest_frame_seq_ = 0;
    }

    worker_thread_ = std::thread(&HikCameraCore::worker_loop, this);

    print_info("Start grabbing");
    return true;
}

void HikCameraCore::worker_loop() {
    int nRet = MV_OK;

    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(handle_, "PayloadSize", &stParam);
    if (MV_OK != nRet) {
        printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
        return;
    }

    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    MV_CC_PIXEL_CONVERT_PARAM stConvertParam = {0};

    std::vector<unsigned char> pData(static_cast<size_t>(stParam.nCurValue) * 3);
    std::vector<unsigned char> pDataForBGR(static_cast<size_t>(stParam.nCurValue) * 3);

    if (pData.empty() || pDataForBGR.empty()) {
        printf("Memory allocation failed!\n");
        return;
    }

    printf("[HikCamera] worker_loop started, payload=%u\n", stParam.nCurValue);

    while (!worker_stop_.load()) {
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        memset(&stConvertParam, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));

        nRet = MV_CC_GetOneFrameTimeout(
            handle_,
            pData.data(),
            stParam.nCurValue * 3,
            &stImageInfo,
            1000
        );

        if (nRet != MV_OK) {
            printf("MV_CC_GetOneFrameTimeout failed! nRet [0x%x]\n", nRet);
            continue;
        }

        stConvertParam.nWidth = stImageInfo.nWidth;
        stConvertParam.nHeight = stImageInfo.nHeight;
        stConvertParam.pSrcData = pData.data();
        stConvertParam.nSrcDataLen = stParam.nCurValue * 3;
        stConvertParam.enSrcPixelType = stImageInfo.enPixelType;
        stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
        stConvertParam.pDstBuffer = pDataForBGR.data();
        stConvertParam.nDstBufferSize = stParam.nCurValue * 3;

        static bool printed_success = false;

        nRet = MV_CC_ConvertPixelType(handle_, &stConvertParam);
        if (MV_OK != nRet) {
            printf("MV_CC_ConvertPixelType failed! nRet [0x%x], width=%d, height=%d, frame_len=%lu, src_pixel_type=%lu\n",
                nRet,
                stImageInfo.nWidth,
                stImageInfo.nHeight,
                (unsigned long)stImageInfo.nFrameLen,
                (unsigned long)stImageInfo.enPixelType);
            continue;
        } else {
            if (!printed_success) {
                printf("[HikCamera] Pixel convert OK. width=%d, height=%d\n",
                    stImageInfo.nWidth,
                    stImageInfo.nHeight);
                printed_success = true;
            }
        }

        cv::Mat srcImage(
            static_cast<int>(stImageInfo.nHeight),
            static_cast<int>(stImageInfo.nWidth),
            CV_8UC3,
            pDataForBGR.data()
        );

        if (image_scale_ > 0.0) {
            cv::resize(
                srcImage,
                srcImage,
                cv::Size(
                    static_cast<int>(srcImage.cols * image_scale_),
                    static_cast<int>(srcImage.rows * image_scale_)
                ),
                0,
                0,
                cv::INTER_LINEAR
            );
        } else {
            printf("Invalid image_scale: %f. Skipping resize.\n", image_scale_);
        }

        FrameData frame;
        frame.width = srcImage.cols;
        frame.height = srcImage.rows;
        frame.channels = 3;
        frame.timestamp = now_sec();
        frame.data.assign(
            srcImage.data,
            srcImage.data + srcImage.total() * srcImage.elemSize()
        );

        {
            std::lock_guard<std::mutex> lk(frame_mutex_);
            latest_frame_ = std::move(frame);
            latest_frame_seq_++;
            width_ = latest_frame_.width;
            height_ = latest_frame_.height;
        }
        frame_cv_.notify_all();
    }

    printf("[HikCamera] worker_loop exiting\n");
}

FrameData HikCameraCore::get_frame(int timeout_ms) {
    if (!opened_ || !grabbing_) {
        throw std::runtime_error("Camera must be opened and started before get_frame()");
    }

    std::unique_lock<std::mutex> lk(frame_mutex_);

    uint64_t start_seq = latest_frame_seq_;
    bool ok = frame_cv_.wait_for(
        lk,
        std::chrono::milliseconds(timeout_ms),
        [&]() { return latest_frame_seq_ > start_seq || !latest_frame_.data.empty(); }
    );

    if (!ok || latest_frame_.data.empty()) {
        throw std::runtime_error("Timed out waiting for a converted frame from worker thread");
    }

    return latest_frame_;
}

void HikCameraCore::stop() {
    {
        std::lock_guard<std::mutex> lock(api_mutex_);
        if (!opened_ || !grabbing_) return;
        worker_stop_.store(true);
    }

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    std::lock_guard<std::mutex> lock(api_mutex_);
    if (grabbing_) {
        int nRet = MV_CC_StopGrabbing(handle_);
        check_mv(nRet, "MV_CC_StopGrabbing");
        grabbing_ = false;
        print_info("Stop grabbing");
    }
}

void HikCameraCore::close() {
    stop();

    std::lock_guard<std::mutex> lock(api_mutex_);

    if (handle_ != nullptr) {
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
    }

    opened_ = false;
    grabbing_ = false;
    worker_stop_.store(false);
    payload_size_ = 0;
    width_ = 0;
    height_ = 0;

    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = FrameData();
        latest_frame_seq_ = 0;
    }
}