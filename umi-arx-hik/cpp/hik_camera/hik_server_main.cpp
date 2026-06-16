#include "hik_camera_core.h"
#include "hik_frame_protocol.h"

#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static volatile std::sig_atomic_t g_stop = 0;

void signal_handler(int) {
    g_stop = 1;
}

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --config <yaml> --shm-name </hik_cam0>\n";
}

int main(int argc, char** argv) {
    std::string config_path;
    std::string shm_name = "/hik_cam0";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--shm-name" && i + 1 < argc) {
            shm_name = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        perror("shm_open");
        return 1;
    }

    if (ftruncate(fd, HIK_SHM_TOTAL_BYTES) != 0) {
        perror("ftruncate");
        close(fd);
        return 1;
    }

    void* mem = mmap(nullptr, HIK_SHM_TOTAL_BYTES,
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    auto* header = reinterpret_cast<HikShmHeader*>(mem);
    auto* image_ptr = reinterpret_cast<unsigned char*>(
        reinterpret_cast<char*>(mem) + sizeof(HikShmHeader));

    std::memset(header, 0, sizeof(HikShmHeader));
    header->magic = HIK_SHM_MAGIC;
    header->seq = 0;
    header->status = 0;

    try {
        HikCameraCore cam(config_path);
        cam.open();
        cam.start();

        std::cout << "[hik_server] started. shm=" << shm_name << std::endl;

        while (!g_stop) {
            FrameData frame = cam.get_frame(3000);

            if (frame.data.empty()) {
                continue;
            }
            if (frame.data.size() > HIK_MAX_IMAGE_BYTES) {
                std::cerr << "[hik_server] frame too large: "
                          << frame.data.size() << std::endl;
                continue;
            }

            // begin write: set seq to odd
            uint64_t seq = header->seq;
            if ((seq % 2) == 0) {
                seq += 1;
            } else {
                seq += 2;
            }
            header->seq = seq;
            __sync_synchronize();

            header->magic = HIK_SHM_MAGIC;
            header->width = static_cast<uint32_t>(frame.width);
            header->height = static_cast<uint32_t>(frame.height);
            header->channels = static_cast<uint32_t>(frame.channels);
            header->data_bytes = static_cast<uint32_t>(frame.data.size());
            header->timestamp = frame.timestamp;
            header->status = 1;

            std::memcpy(image_ptr, frame.data.data(), frame.data.size());

            __sync_synchronize();
            header->seq = seq + 1;  // even => stable
        }

        cam.stop();
        cam.close();
    } catch (const std::exception& e) {
        std::cerr << "[hik_server] ERROR: " << e.what() << std::endl;
    }

    munmap(mem, HIK_SHM_TOTAL_BYTES);
    close(fd);
    // 不 unlink，方便客户端先于服务端启动/重启
    return 0;
}