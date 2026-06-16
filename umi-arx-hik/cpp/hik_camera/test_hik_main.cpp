#include <iostream>
#include "hik_camera_core.h"

int main() {
    try {
        std::string cfg = "/home/wzm/umi-arx-hik/config/left_camera_trigger.yaml";

        HikCameraCore cam(cfg);
        std::cout << "opening..." << std::endl;
        cam.open();

        std::cout << "starting..." << std::endl;
        cam.start();

        std::cout << "waiting frame..." << std::endl;
        FrameData frame = cam.get_frame(5000);

        std::cout << "success: "
                  << frame.width << " "
                  << frame.height << " "
                  << frame.channels << " "
                  << frame.timestamp << " "
                  << frame.data.size() << std::endl;

        cam.stop();
        cam.close();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}