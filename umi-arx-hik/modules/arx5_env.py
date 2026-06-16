from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from modules.arx5_controller import Arx5Controller
from peripherals.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from peripherals.multi_hik_camera import MultiHikCamera
from modules.timestamp_accumulator import TimestampActionAccumulator, ObsAccumulator
from utils.cv_util import draw_predefined_mask
from peripherals.multi_camera_visualizer import MultiCameraVisualizer
from modules.replay_buffer import ReplayBuffer
from utils.cv2_util import get_image_transform, optimal_row_cols
from utils.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from utils.interpolation_util import get_interp1d, PoseInterpolator


class Arx5Env:
    def __init__(
        self,
        # required params
        output_dir,
        robots_config,  # list of dict[{robot_type: 'ur5', robot_ip: XXX, obs_latency: 0.0001, action_latency: 0.1, tcp_offset: 0.21}]
        # env params
        frequency=20,
        # obs
        obs_image_resolution=(224, 224),
        max_obs_buffer_size=60,
        obs_float32=False,
        camera_reorder=None,
        no_mirror=False,
        fisheye_converter=None,
        mirror_swap=False,
        # this latency compensates receive_timestamp
        # all in seconds
        camera_obs_latency=0.125,
        # all in steps (relative to frequency)
        camera_down_sample_steps=1,
        robot_down_sample_steps=1,
        # all in steps (relative to frequency)
        camera_obs_horizon=2,
        robot_obs_horizon=2,
        # action
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        init_joints=False,
        # vis params
        enable_multi_cam_vis=True,
        multi_cam_vis_resolution=(960, 960),
        # camera backend
        camera_backend="uvc",
        camera_capture_fps=60,
        hik_server_bin=None,
        hik_config_paths=None,
        hik_shm_names=None,        
        # shared memory
        shm_manager=None,
    ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        self.camera_backend = camera_backend
        self.camera_capture_fps = camera_capture_fps
        self.hik_server_bin = hik_server_bin
        self.hik_config_paths = hik_config_paths
        self.hik_shm_names = hik_shm_names

        if self.camera_backend == "uvc":
            # Find and reset all Elgato capture cards.
            # Required to workaround a firmware bug.
            reset_all_elgato_devices()

            # Wait for all v4l cameras to be back online
            time.sleep(0.1)
            v4l_paths = get_sorted_v4l_paths()
            if camera_reorder is not None:
                paths = [v4l_paths[i] for i in camera_reorder]
                v4l_paths = paths

        elif self.camera_backend == "hik":
            if self.hik_server_bin is None:
                raise ValueError("hik_server_bin must be provided when camera_backend='hik'")
            if self.hik_config_paths is None or len(self.hik_config_paths) == 0:
                raise ValueError("hik_config_paths must be provided when camera_backend='hik'")

            v4l_paths = list(self.hik_config_paths)

        else:
            raise ValueError("Unknown camera_backend: {}".format(self.camera_backend))

        # compute resolution for vis
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=4 / 3,
            max_resolution=multi_cam_vis_resolution,
        )

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Other capture card records at 720p 60fps
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        for path in v4l_paths:
            if self.camera_backend == "hik":
                res = (1280, 1024)
                fps = self.camera_capture_fps
                buf = 1
                bit_rate = 3000 * 1000

                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224, 224, 3), dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask,
                        color=(0, 0, 0),
                        mirror=True,
                        gripper=False,
                        finger=False,
                    )
                    is_mirror = mirror_mask[..., 0] == 0

                def tf_hik(data, input_res=res):
                    img = data["color"]
                    if fisheye_converter is None:
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution,
                            # Hik server already outputs RGB
                            bgr_to_rgb=False,
                        )
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:, ::-1, :][is_mirror]
                        img = draw_predefined_mask(
                            img,
                            color=(0, 0, 0),
                            mirror=no_mirror,
                            gripper=True,
                            finger=False,
                            use_aa=True,
                        )
                    else:
                        img = fisheye_converter.forward(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data["color"] = img
                    return data

                transform.append(tf_hik)

            elif "Cam_Link_4K" in path:
                res = (3840, 2160)
                fps = 30
                buf = 3
                bit_rate = 6000 * 1000

                def tf4k(data, input_res=res):
                    img = data["color"]
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution,
                        # obs output rgb
                        bgr_to_rgb=True,
                    )
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data["color"] = img
                    return data

                transform.append(tf4k)
            else:
                res = (1920, 1080)
                fps = 60
                buf = 1
                bit_rate = 3000 * 1000

                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224, 224, 3), dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask,
                        color=(0, 0, 0),
                        mirror=True,
                        gripper=False,
                        finger=False,
                    )
                    is_mirror = mirror_mask[..., 0] == 0

                def tf(data, input_res=res):
                    img = data["color"]
                    if fisheye_converter is None:
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution,
                            # obs output rgb
                            bgr_to_rgb=True,
                        )
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:, ::-1, :][is_mirror]
                        img = draw_predefined_mask(
                            img,
                            color=(0, 0, 0),
                            mirror=no_mirror,
                            gripper=True,
                            finger=False,
                            use_aa=True,
                        )
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[..., ::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data["color"] = img
                    return data

                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            input_pix_fmt = "rgb24" if self.camera_backend == "hik" else "bgr24"
            video_recorder.append(
                VideoRecorder.create_hevc_nvenc(
                    fps=fps, input_pix_fmt=input_pix_fmt, bit_rate=bit_rate
                )
            )

            def vis_tf(data, input_res=res):
                img = data["color"]
                # Hik path is already RGB; vis path should keep color order unchanged
                f = get_image_transform(
                    input_res=input_res,
                    output_res=(rw, rh),
                    bgr_to_rgb=False,
                )
                img = f(img)
                data["color"] = img
                return data

            vis_transform.append(vis_tf)

        if self.camera_backend == "uvc":
            camera = MultiUvcCamera(
                dev_video_paths=v4l_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=capture_fps,
                # send every frame immediately after arrival
                # ignores put_fps
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                cap_buffer_size=cap_buffer_size,
                transform=transform,
                # vis_transform=vis_transform, # TODO: vis_transform doesn't work?
                video_recorder=video_recorder,
                verbose=False,
            )
        elif self.camera_backend == "hik":
            camera = MultiHikCamera(
                shm_manager=shm_manager,
                server_bin=self.hik_server_bin,
                config_paths=self.hik_config_paths,
                shm_names=self.hik_shm_names,
                resolution=resolution,
                capture_fps=capture_fps,
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                transform=transform,
                vis_transform=vis_transform,
                video_recorder=video_recorder,
                verbose=False,
            )
        else:
            raise ValueError("Unknown camera_backend: {}".format(self.camera_backend))

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera, row=row, col=col, rgb_to_bgr=False
            )

        robots: List[Arx5Controller] = list()
        for rc in robots_config:
            this_robot = Arx5Controller(  # TODO:
                shm_manager=shm_manager,
                robot_ip=rc["robot_ip"],
                robot_port=rc["robot_port"],
                frequency=200,
                verbose=True,
            )
            robots.append(this_robot)

        self.camera = camera

        self.robots = robots
        self.robots_config = robots_config

        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self.last_time_step = 0

    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready_flag = self.camera.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
        return ready_flag

    def start(self, wait=True):
        self.camera.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        assert self.is_ready
        # get data
        # 60 Hz, camera_calibrated_timestamp
        k = (
            math.ceil(
                self.camera_obs_horizon
                * self.camera_down_sample_steps
                * (self.camera_capture_fps / self.frequency)
            )
            + 2
        )  # here 2 is adjustable, typically 1 should be enough
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)
        self.last_camera_data = self.camera.get(k=k, out=self.last_camera_data)

        # both have more than n_obs_steps data
        last_robots_data = list()
        # 125/500 hz, robot_receive_timestamp
        for robot in self.robots:
            last_robots_data.append(robot.get_all_state())

        # select align_camera_idx
        num_obs_cameras = len(self.robots)
        align_camera_idx = None
        running_best_error = np.inf

        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]["timestamp"][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if (
                        self.last_camera_data[other_camera_idx]["timestamp"][
                            other_timestep_idx
                        ]
                        < this_timestamp
                    ):
                        this_error += (
                            this_timestamp
                            - self.last_camera_data[other_camera_idx]["timestamp"][
                                other_timestep_idx
                            ]
                        )
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        last_timestamp = self.last_camera_data[align_camera_idx]["timestamp"][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1]
            * self.camera_down_sample_steps
            * dt
        )
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 120 and camera_idx != 3:
                #     print('ERROR!!!  ', camera_idx, len(this_timestamps), nn_idx, (this_timestamps - t)[nn_idx-1: nn_idx+2])
                this_idxs.append(nn_idx)
            # remap key
            camera_obs[f"camera{camera_idx}_rgb"] = value["color"][this_idxs]

        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)

        # include camera timesteps
        obs_data["timestamp"] = camera_obs_timestamps

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt
        )
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data["robot_timestamp"], x=last_robot_data["ActualTCPPose"]
            )
            gripper_pos_interpolator = get_interp1d(
                t=last_robot_data["robot_timestamp"],
                x=last_robot_data["gripper_position"][..., None],
            )
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            gripper_pos = gripper_pos_interpolator(robot_obs_timestamps)
            robot_obs = {
                f"robot{robot_idx}_eef_pos": robot_pose[..., :3],
                f"robot{robot_idx}_eef_rot_axis_angle": robot_pose[..., 3:],
                f"robot{robot_idx}_gripper_width": gripper_pos,
            }

            # update obs_data
            obs_data.update(robot_obs)

        # accumulate obs
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f"robot{robot_idx}_eef_pose": last_robot_data["ActualTCPPose"],
                        f"robot{robot_idx}_joint_pos": last_robot_data["ActualQ"],
                        f"robot{robot_idx}_joint_vel": last_robot_data["ActualQd"],
                        f"robot{robot_idx}_gripper_width": last_robot_data[
                            "gripper_position"
                        ],
                    },
                    timestamps=last_robot_data["robot_timestamp"],
                )

        return obs_data

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        compensate_latency=False,
        dynamic_latency=False,
    ):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        assert new_actions.shape[1] // len(self.robots) == 7
        assert new_actions.shape[1] % len(self.robots) == 0

        # schedule waypoints

        if not dynamic_latency:
            for i in range(len(new_actions)):
                for robot_idx, (robot, rc) in enumerate(
                    zip(self.robots, self.robots_config)
                ):
                    r_latency = (
                        rc["robot_action_latency"] if compensate_latency else 0.0
                    )
                    r_actions = new_actions[i, 7 * robot_idx + 0 : 7 * robot_idx + 6]
                    g_actions = new_actions[i, 7 * robot_idx + 6]
                    robot.schedule_waypoint(
                        pose=r_actions,
                        gripper_pos=g_actions,
                        target_time=new_timestamps[i] - r_latency,
                    )
        else:
            for robot_idx, (robot, rc) in enumerate(
                zip(self.robots, self.robots_config)
            ):
                for i in range(len(new_actions)):
                    r_actions = new_actions[i, 7 * robot_idx + 0 : 7 * robot_idx + 6]
                    g_actions = new_actions[i, 7 * robot_idx + 6]
                    robot.add_waypoint(
                        pose=r_actions,
                        gripper_pos=g_actions,
                        target_time=new_timestamps[i],
                    )
                robot.update_trajectory()

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)

    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency
        )
        print(f"Episode {episode_id} started!")

    def end_episode(self):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float("inf")
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = (
                    np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]
                    + 1
                )

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    "timestamp": timestamps,
                    "action": actions[:n_steps],
                }
                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_eef_pose"
                            ]
                        ),
                        x=np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_eef_pose"]
                        ),
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f"robot{robot_idx}_eef_pos"] = robot_pose[:, :3]
                    episode[f"robot{robot_idx}_eef_rot_axis_angle"] = robot_pose[:, 3:]
                    joint_pos_interpolator = get_interp1d(
                        np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_joint_pos"
                            ]
                        ),
                        np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_joint_pos"]
                        ),
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_joint_vel"
                            ]
                        ),
                        np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_joint_vel"]
                        ),
                    )
                    episode[f"robot{robot_idx}_joint_pos"] = joint_pos_interpolator(
                        timestamps
                    )
                    episode[f"robot{robot_idx}_joint_vel"] = joint_vel_interpolator(
                        timestamps
                    )

                    gripper_interpolator = get_interp1d(
                        t=np.array(
                            self.obs_accumulator.timestamps[
                                f"robot{robot_idx}_gripper_width"
                            ]
                        ),
                        x=np.array(
                            self.obs_accumulator.data[f"robot{robot_idx}_gripper_width"]
                        ),
                    )
                    episode[f"robot{robot_idx}_gripper_width"] = gripper_interpolator(
                        timestamps
                    )

                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")

            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f"Episode {episode_id} dropped!")