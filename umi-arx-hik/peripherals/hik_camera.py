import copy
import enum
import mmap
import os
import struct
import subprocess
import time
from typing import Optional, Callable, Dict

import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from peripherals.video_recorder import VideoRecorder


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


class HikCamera(mp.Process):
    MAX_PATH_LENGTH = 4096
    SHM_MAGIC = 0x48494B31
    HEADER_STRUCT = struct.Struct("<IIIIIIQd")  # magic,w,h,c,data_bytes,status,seq,timestamp
    HEADER_SIZE = HEADER_STRUCT.size
    MAX_IMAGE_BYTES = 5 * 1024 * 1024
    TOTAL_BYTES = HEADER_SIZE + MAX_IMAGE_BYTES

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        server_bin: str,
        config_path: str,
        shm_name: str = "/hik_cam0",
        resolution=(1280, 1024),
        capture_fps=60,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        receive_latency=0.0,
        num_threads=2,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        resolution = tuple(resolution)
        shape = resolution[::-1]

        examples = {
            "color": np.empty(shape=shape + (3,), dtype=np.uint8),
            "camera_capture_timestamp": 0.0,
            "camera_receive_timestamp": 0.0,
            "timestamp": 0.0,
            "step_idx": 0,
        }

        vis_examples = copy.deepcopy(examples)
        if vis_transform is not None:
            # 先给个常见可视化大小；真正 put 时由 vis_transform 输出决定
            vis_examples["color"] = np.empty(shape=(720, 960, 3), dtype=np.uint8)
        else:
            vis_examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)

        ring_buffer_example = copy.deepcopy(examples)
        if transform is not None:
            ring_buffer_example = transform(dict(ring_buffer_example))

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=ring_buffer_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        self.vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=vis_examples,
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        queue_examples = {
            "cmd": Command.RESTART_PUT.value,
            "put_start_time": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=queue_examples,
            buffer_size=128,
        )

        if video_recorder is None:
            video_recorder = VideoRecorder.create_hevc_nvenc(
                shm_manager=shm_manager,
                fps=capture_fps,
                input_pix_fmt="rgb24",
                bit_rate=3000 * 1000,
            )

        self.shm_manager = shm_manager
        self.server_bin = server_bin
        self.config_path = config_path
        self.shm_name = shm_name
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.num_threads = num_threads

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

        self.server_proc = None
        self.put_start_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time

        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape + (3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager,
            data_example=data_example,
        )

        # 启动系统环境下的 server
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/opt/MVS/lib/64:" + env.get("LD_LIBRARY_PATH", "")
        self.server_proc = subprocess.Popen(
            [
                self.server_bin,
                "--config", self.config_path,
                "--shm-name", self.shm_name,
            ],
            env=env,
        )

        super().start()

        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.video_recorder.stop()
        self.stop_event.set()

        if self.server_proc is not None:
            self.server_proc.terminate()

        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        self.video_recorder.start_wait()

    def end_wait(self):
        self.join()
        self.video_recorder.end_wait()

        if self.server_proc is not None:
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                self.server_proc.wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float = -1):
        if len(video_path.encode("utf-8")) > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long")
        self.command_queue.put({
            "cmd": Command.START_RECORDING.value,
            "video_path": video_path,
            "recording_start_time": start_time,
        })

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put({
            "cmd": Command.RESTART_PUT.value,
            "put_start_time": start_time,
        })

    def _read_latest_frame(self, mm, last_seq):
        """
        读取最新稳定帧。
        返回: (result_dict_or_None, new_last_seq)
        """
        mm.seek(0)
        raw = mm.read(self.HEADER_SIZE)
        if len(raw) != self.HEADER_SIZE:
            return None, last_seq

        magic, w, h, c, data_bytes, status, seq1, ts = self.HEADER_STRUCT.unpack(raw)

        if magic != self.SHM_MAGIC or status != 1:
            return None, last_seq

        # odd -> writer in progress
        if seq1 % 2 == 1:
            return None, last_seq

        # 没有新帧
        if seq1 == last_seq:
            return None, last_seq

        if data_bytes == 0 or data_bytes > self.MAX_IMAGE_BYTES:
            return None, last_seq

        mm.seek(self.HEADER_SIZE)
        img_bytes = mm.read(data_bytes)
        if len(img_bytes) != data_bytes:
            return None, last_seq

        # 再读一次头，确认 seq 没变
        mm.seek(0)
        raw2 = mm.read(self.HEADER_SIZE)
        if len(raw2) != self.HEADER_SIZE:
            return None, last_seq
        _, w2, h2, c2, data_bytes2, status2, seq2, ts2 = self.HEADER_STRUCT.unpack(raw2)

        if seq1 != seq2 or seq2 % 2 == 1:
            return None, last_seq

        if (w, h, c, data_bytes, status) != (w2, h2, c2, data_bytes2, status2):
            return None, last_seq

        img = np.frombuffer(img_bytes, dtype=np.uint8).copy()
        img = img.reshape(h, w, c)

        result = {
            "color": img,
            "camera_capture_timestamp": ts,
            "camera_receive_timestamp": ts,
            "timestamp": ts - self.receive_latency,
        }
        return result, seq2

    def run(self):
        pid = os.getpid()
        try:
            os.sched_setaffinity(pid, [5])
        except Exception:
            pass

        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        shm_path = "/dev/shm/" + self.shm_name.lstrip("/")
        t0 = time.time()
        while not os.path.exists(shm_path):
            if self.stop_event.is_set():
                return
            if time.time() - t0 > 10:
                raise RuntimeError(f"Timeout waiting for shared memory file: {shm_path}")
            time.sleep(0.05)

        with open(shm_path, "r+b", buffering=0) as f:
            mm = mmap.mmap(f.fileno(), self.TOTAL_BYTES, access=mmap.ACCESS_READ)

            last_seq = 0
            put_idx = 0
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            first_frame_published = False

            while not self.stop_event.is_set():
                try:
                    commands = self.command_queue.get_all()

                    # commands is a dict of batched fields, not a list of dicts
                    n_cmd = len(commands["cmd"])
                    for i in range(n_cmd):
                        cmd = commands["cmd"][i]

                        if cmd == Command.RESTART_PUT.value:
                            put_idx = 0
                            put_start_time = commands["put_start_time"][i]

                        elif cmd == Command.START_RECORDING.value:
                            video_path = commands["video_path"][i]
                            if isinstance(video_path, np.ndarray):
                                video_path = video_path.item()
                            if isinstance(video_path, bytes):
                                video_path = video_path.decode("utf-8")
                            video_path = str(video_path).rstrip("\x00")

                            self.video_recorder.start_recording(
                                video_path,
                                start_time=commands["recording_start_time"][i],
                            )

                        elif cmd == Command.STOP_RECORDING.value:
                            self.video_recorder.stop_recording()

                except Empty:
                    pass
                except Exception as e:
                    print("[HikCamera] command processing error:", repr(e), flush=True)

                result, last_seq = self._read_latest_frame(mm, last_seq)
                if result is None:
                    time.sleep(0.002)
                    continue

                result["step_idx"] = put_idx
                put_idx += 1

                data = result
                if self.transform is not None:
                    data = self.transform(dict(data))
                self.ring_buffer.put(data)

                vis_data = result
                if self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(result))
                self.vis_ring_buffer.put(vis_data)

                if self.video_recorder.is_ready():
                    try:
                        rec_data = dict(result)
                        if self.recording_transform is not None:
                            rec_data = self.recording_transform(dict(result))

                        rec_ts = rec_data.get("timestamp", None)

                        # rec_ts may be a scalar, numpy scalar, or length-1 array
                        if isinstance(rec_ts, np.ndarray):
                            if rec_ts.size == 0:
                                rec_ts = None
                            else:
                                rec_ts = float(rec_ts.reshape(-1)[0])

                        if rec_ts is None:
                            print("[HikCamera] skip recording frame because timestamp is None", flush=True)
                        else:
                            # make sure timestamp is explicitly passed through
                            rec_data["timestamp"] = rec_ts
                            self.video_recorder.write_frame(rec_data["color"], frame_time=rec_ts)

                    except Exception as e:
                        print("[HikCamera] video recording error:", repr(e), flush=True)

                # 关键修复：首帧真正进入 ring buffer 后再 ready
                if not first_frame_published:
                    self.ready_event.set()
                    first_frame_published = True

            mm.close()