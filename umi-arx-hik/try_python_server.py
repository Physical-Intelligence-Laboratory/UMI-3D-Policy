import time
from multiprocessing.managers import SharedMemoryManager
from peripherals.hik_camera import HikCamera

with SharedMemoryManager() as shm_manager:
    cam = HikCamera(
        shm_manager=shm_manager,
        server_bin="/home/wzm/umi-arx-hik/cpp/hik_camera/build_sys/hik_server",
        config_path="/home/wzm/umi-arx-hik/config/left_camera_trigger.yaml",
        shm_name="/hik_cam0",
        resolution=(1280, 1024),
        capture_fps=60,
    )
    cam.start(wait=True)

    print("ready:", cam.is_ready)

    data = cam.get(k=1)
    print(data["color"].shape)
    print(data["timestamp"])

    cam.stop(wait=True)
