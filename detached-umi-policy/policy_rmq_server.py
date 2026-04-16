import sys
import os
import time
from typing import Any, Optional
import click
import numpy as np
import torch
import dill
import hydra
import zmq
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.cv_util import draw_predefined_mask
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action, get_real_umi_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
import omegaconf
import traceback
from robotmq import RMQServer, serialize, deserialize

def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)

def qconjugate(q: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qmult(q1: npt.NDArray[Any], q2: npt.NDArray[Any]) -> npt.NDArray[Any]:
    q = np.array(
        [
            q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
        ]
    )

    return q

def to_xyzw(wxyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if wxyz.ndim == 1:
        return np.concatenate([wxyz[1:], wxyz[0:1]])
    elif wxyz.ndim == 2:
        return np.concatenate([wxyz[:, 1:], wxyz[:, 0:1]], axis=1)
    else:
        raise ValueError("wxyz must be a 1D or 2D array")


def to_wxyz(xyzw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if xyzw.ndim == 1:
        return np.concatenate([xyzw[3:], xyzw[:3]])
    elif xyzw.ndim == 2:
        return np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
    else:
        raise ValueError("xyzw must be a 1D or 2D array")

def get_absolute_pose(
    init_pose_xyz_wxyz: npt.NDArray[Any],
    relative_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The new pose is in the same frame of reference as the initial pose"""
    new_pose_xyz_wxyz = np.zeros(7, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz = np.zeros(4, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz[1:] = relative_pose_xyz_wxyz[:3]
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pos_in_world_frame_as_quat_wxyz = qmult(
        qmult(init_pose_xyz_wxyz[3:], relative_pos_in_init_frame_as_quat_wxyz),
        init_rot_qinv,
    )
    new_pose_xyz_wxyz[:3] = (
        init_pose_xyz_wxyz[:3] + relative_pos_in_world_frame_as_quat_wxyz[1:]
    )
    quat = qmult(init_pose_xyz_wxyz[3:], relative_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    new_pose_xyz_wxyz[3:] = quat
    return new_pose_xyz_wxyz


def get_relative_pose(
    new_pose_xyz_wxyz: npt.NDArray[Any],
    init_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The two poses are in the same frame of reference"""
    relative_pose_xyz_wxyz = np.zeros(7, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz = np.zeros(4, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz[1:] = (
        new_pose_xyz_wxyz[:3] - init_pose_xyz_wxyz[:3]
    )
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pose_xyz_wxyz[:3] = qmult(
        qmult(init_rot_qinv, relative_pos_in_world_frame_as_quat_wxyz),
        init_pose_xyz_wxyz[3:],
    )[1:]
    quat = qmult(init_rot_qinv, new_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    relative_pose_xyz_wxyz[3:] = quat
    return relative_pose_xyz_wxyz

class PolicyInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int, device: str, use_shared_memory: bool):
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
        with open(cfg_path, 'w') as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"Exported config to {cfg_path}")
        print(f"Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.policy._target_}, model_name: {self.cfg.policy.obs_encoder.model_name}")
        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.get_class_start_time = time.monotonic()

        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy:BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")
        self.policy.num_inference_steps = 16
        
        obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('obs_pose_rep', obs_pose_rep)
        print('action_pose_repr', action_pose_repr)
        
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.ip = ip
        self.port = port

        self.rmq_server = RMQServer(server_name="umi_policy", server_endpoint=f"tcp://{ip}:{port}")
        if use_shared_memory:
            self.rmq_server.add_shared_memory_topic("policy_inference", message_remaining_time_s=10, shared_memory_size_gb=1.0)
        else:
            self.rmq_server.add_topic("policy_inference", message_remaining_time_s=10)

        self.rmq_server.add_topic("reset", message_remaining_time_s=10)

        # States
        self.episode_start_pose_pos_rotvec: Optional[npt.NDArray[np.float64]] = None
    
    def reset(self):
        self.episode_start_pose_pos_rotvec = None

    def predict_action(self, obs_dict_np: dict[str, Any]):
        """
        Currently only support single robot
        obs_dict_np: dict # All absolute pose
            "robot{i}_eef_xyz_wxyz": (N, 7)
            "robot{i}_gripper_width": (N, 1)
            "robot{i}_wrist_camera": (N, H, W, 3)
        """
        if self.episode_start_pose_pos_rotvec is None:
            pos = obs_dict_np["robot0_eef_xyz_wxyz"][0, :3]
            rotvec = R.from_quat(to_xyzw(obs_dict_np["robot0_eef_xyz_wxyz"][0, 3:])).as_rotvec()
            self.episode_start_pose_pos_rotvec = np.concatenate([pos, rotvec])

        assert self.episode_start_pose_pos_rotvec is not None
        
        eef_xyz_wxyz = obs_dict_np.pop("robot0_eef_xyz_wxyz")
        # eef_xyz_wxyz_wrt_start = get_relative_pose(eef_xyz_wxyz, self.episode_start_pose)
        obs_dict_np["robot0_eef_pos"] = eef_xyz_wxyz[:3]
        obs_dict_np["robot0_eef_rot_axis_angle"] = R.from_quat(to_xyzw(eef_xyz_wxyz[3:])).as_rotvec()
        # obs_dict_np["robot0_eef_rot_axis_angle_wrt_start"] = R.from_quat(to_xyzw(eef_xyz_wxyz_wrt_start[3:])).as_rotvec()
        obs_dict_np["camera0_rgb"] = draw_predefined_mask(
            obs_dict_np.pop("robot0_wrist_camera"),
            color=(0, 0, 0),
            mirror=True,
            gripper=True,
            finger=False,
            use_aa=True,
        )
        obs_dict_np["robot0_gripper_width"] = obs_dict_np.pop("robot0_gripper_width")

        """
        obs_dict_np: dict
            "robot{i}_eef_pos": (N, 3)
            "robot{i}_eef_rot_axis_angle": (N, 6) # is actually represented by rotation6d
            "robot{i}_eef_rot_axis_angle_wrt_start": (N, 6) # is actually represented by rotation6d
            "robot{i}_gripper_width": (N, 1)
            "camera{i}_rgb": (M, 3, H, W) # M: rgb_obs_horizon
        """
        with torch.no_grad():
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
            del result
            del obs_dict
        return action
    
    def run_node(self):
        while True:
            raw_data, topic = self.rmq_server.wait_for_request(timeout_s=1)
            if topic == "reset":
                self.reset()
                self.rmq_server.reply_request(topic="reset", data=serialize("OK"))
                print("Done policy reset")
                continue
            obs_dict_np = deserialize(raw_data)
            try:
                assert topic == "policy_inference"
                start_time = time.monotonic()
                action = self.predict_action(obs_dict_np)
                print(f'Inference time: {time.monotonic() - start_time:.3f} s')
            except Exception as e:
                err_str = echo_exception()
                print(f'Error: {err_str}')
                action = err_str
            send_start_time = time.monotonic()
            self.rmq_server.reply_request(topic="policy_inference", data=serialize(action))
            print(f'Send time: {time.monotonic() - send_start_time:.3f} s')
    
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--ip', default="0.0.0.0")
@click.option('--port', default=8766, help="Port to listen on")
@click.option('--device', default="cuda", help="Device to run on")
@click.option('--use_shared_memory', is_flag=True, default=False, help="Use shared memory for communication")
def main(input, ip, port, device, use_shared_memory):
    node = PolicyInferenceNode(input, ip, port, device, use_shared_memory)
    node.run_node()
                

if __name__ == '__main__':
    main()