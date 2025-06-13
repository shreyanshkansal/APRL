import os

import numpy as np
from .sim_robot import RailSimWalkerDMControl, SIM_ASSET_DIR
from functools import cached_property

#The go2 xml file is gotten from Unitree Mujoco with slight modifications
_Go2_XML_PATH = os.path.join(SIM_ASSET_DIR, 'robot_assets', 'go2', 'go2.xml')
class Go2SimWalker(RailSimWalkerDMControl):
    # _INIT_QPOS = np.asarray([0.0, 0.9, -1.8] * 4)
    # _QPOS_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)

    def __init__(self, Kp=60, Kd=6, action_interpolation : bool = True, limit_action_range : float = 1.0, *args, **kwargs):
        RailSimWalkerDMControl.__init__(
            self,
            XML_FILE = _Go2_XML_PATH,
            Kp = Kp,
            Kd = Kd,
            action_interpolation = action_interpolation,
            limit_action_range = limit_action_range,
            *args,
            **kwargs
        )
        RailSimWalkerDMControl.control_timestep = 0.05
        RailSimWalkerDMControl.control_subtimestep = 0.002

    #the following values are gotten from the xml file
    @cached_property
    def joint_qpos_init(self) -> np.ndarray:
        
        return np.array([
            0.0, 0.9, -1.8,   # FL
            0.0, 0.9, -1.8,   # FR
            0.0, 0.9, -1.8,   # RL
            0.0, 0.9, -1.8    # RR
        ])

    @cached_property
    def joint_qpos_sitting(self) -> np.ndarray:
        return np.array([
            0.0, 1.3, -2.3,   # FL
            0.0, 1.3, -2.3,   # FR
            0.0, 1.3, -2.3,   # RL
            0.0, 1.3, -2.3    # RR
        ])
    
    @cached_property
    def joint_qpos_offset(self) -> np.ndarray:
        return np.array([0.2, 0.4, 0.4] * 4)

    @cached_property
    def joint_qpos_mins(self) -> np.ndarray:
        return np.asarray([
            -1.0472, -1.5708, -2.7227,  # FL (front left)
            -1.0472, -1.5708, -2.7227,  # FR (front right)
            -1.0472, -0.5236, -2.7227,  # RL (rear left)
            -1.0472, -0.5236, -2.7227,  # RR (rear right)
        ])

    @cached_property
    def joint_qpos_maxs(self) -> np.ndarray:
        return np.asarray([
            1.0472, 3.4907, -0.83776,   # FL (front left)
            1.0472, 3.4907, -0.83776,   # FR (front right)
            1.0472, 4.5379, -0.83776,   # RL (rear left)
            1.0472, 4.5379, -0.83776,   # RR (rear right)
        ])

