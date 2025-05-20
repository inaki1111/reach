import torch
from differential_ik_ros import DifferentialIKControllerROS

# 1) Crea el controlador
cfg = {
  "command_type": "pose",
  "use_relative_mode": False,
  "ik_method": "dls",
  "ik_params": {"lambda_val": 0.01}
}
controller = DifferentialIKControllerROS(**cfg, device="cpu")

# 2) Cada paso de control:
#    - ee_pos, ee_quat: tensors (1,3) y (1,4)
#    - jacobian: tensor (1,6,6)
#    - joint_pos: tensor (1,6)
#    - command: tensor (1,7) → [x,y,z,qx,qy,qz,qw]
controller.reset()
controller.set_command(command, ee_pos, ee_quat)
q_target = controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
