# differential_ik_ros.py

import torch
from typing import Literal, Optional

def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of quaternion(s) q of shape (...,4) in (w,x,y,z) format."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply a ⊗ b, both (...,4) wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    return torch.stack((w,x,y,z), dim=-1)

def quaternion_to_axis_angle(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert unit quaternion q (N×4) to (axis N×3, angle N).
    Assumes q is normalized.
    """
    # clamp w to [-1,1]
    w = q[...,0].clamp(-1+1e-7, 1-1e-7)
    angle = 2 * torch.acos(w)                # N
    s = torch.sqrt(1 - w*w)                  # N, sin(angle/2)
    # if s small, axis arbitrary
    axis = torch.where(
        s.unsqueeze(-1) < 1e-7,
        torch.tensor([1.0,0.0,0.0], device=q.device),
        q[...,1:] / s.unsqueeze(-1)
    )
    return axis, angle

def compute_pose_error(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_pos_des: torch.Tensor,
    ee_quat_des: torch.Tensor,
    rot_error_type: Literal["axis_angle"] = "axis_angle"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given current (pos, quat) and desired (pos, quat), compute:
      - position_error (N×3)
      - orientation_error (N×3) as axis*angle if axis_angle.
    """
    pos_err = ee_pos_des - ee_pos  # N×3

    # quaternion error q_e = q_des ⊗ inv(q)
    q_err = quaternion_multiply(ee_quat_des, quaternion_inverse(ee_quat))
    if rot_error_type == "axis_angle":
        axis, angle = quaternion_to_axis_angle(q_err)
        rot_err = axis * angle.unsqueeze(-1)  # N×3
    else:
        raise ValueError(f"rot_error_type '{rot_error_type}' not supported")

    return pos_err, rot_err

class DifferentialIKControllerROS:
    """
    Port of IsaacLab's DifferentialIKController, but standalone.
    """
    def __init__(
        self,
        command_type: Literal["position","pose"] = "pose",
        use_relative_mode: bool = False,
        ik_method: Literal["pinv","svd","trans","dls"] = "dls",
        ik_params: Optional[dict] = None,
        device: torch.device | str = "cpu"
    ):
        self.command_type     = command_type
        self.use_relative_mode = use_relative_mode
        self.ik_method        = ik_method
        self.ik_params        = ik_params or {}
        self.device           = torch.device(device)

        # buffers
        self._command    = None          # Tensor of shape (N,3|6|7)
        self.ee_pos_des  = None          # (N,3)
        self.ee_quat_des = None          # (N,4)

    @property
    def action_dim(self) -> int:
        if self.command_type == "position":
            return 3
        elif self.command_type == "pose" and self.use_relative_mode:
            return 6
        else:
            return 7

    def reset(self):
        self._command = None
        self.ee_pos_des = None
        self.ee_quat_des = None

    def set_command(
        self,
        command: torch.Tensor,
        ee_pos: Optional[torch.Tensor]  = None,
        ee_quat: Optional[torch.Tensor] = None
    ):
        """
        command: (N,3|6|7)
        ee_pos, ee_quat: only needed if use_relative_mode=True
        """
        cmd = command.to(self.device)
        N = cmd.shape[0]
        # init buffers
        self._command = cmd
        # decide ee_pos_des / ee_quat_des
        if self.command_type == "position":
            if ee_quat is None:
                raise ValueError("ee_quat required for position mode")
            ee_quat = ee_quat.to(self.device)
            if self.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("ee_pos required for position_rel")
                ee_pos = ee_pos.to(self.device)
                self.ee_pos_des  = ee_pos + cmd
                self.ee_quat_des = ee_quat
            else:
                self.ee_pos_des  = cmd
                self.ee_quat_des = ee_quat
        else:  # "pose"
            if self.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError("ee_pos/qu required for pose_rel")
                ee_pos  = ee_pos.to(self.device)
                ee_quat = ee_quat.to(self.device)
                # apply delta: cmd[:,:3] pos delta; cmd[:,3:7] delta-quat
                delta_pos  = cmd[:, :3]
                delta_quat = cmd[:, 3:7]
                # new pos = ee_pos + delta_pos
                self.ee_pos_des = ee_pos + delta_pos
                # new quat = delta_quat ⊗ ee_quat
                self.ee_quat_des = quaternion_multiply(delta_quat, ee_quat)
            else:
                # absolute pose
                self.ee_pos_des  = cmd[:, :3]
                self.ee_quat_des = cmd[:, 3:7]

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        ee_pos:   (N,3)
        ee_quat:  (N,4)
        jacobian: (N,6,num_joints)
        joint_pos:(N,num_joints)
        returns:  (N,num_joints) target joint positions
        """
        N, num_j = joint_pos.shape

        if "position" in self.command_type:
            pos_err = self.ee_pos_des - ee_pos  # N×3
            Jp = jacobian[:, 0:3, :]             # N×3×nj
            delta_q = self._compute_delta_joint_pos(pos_err, Jp)
        else:
            p_err, r_err = compute_pose_error(
                ee_pos, ee_quat,
                self.ee_pos_des, self.ee_quat_des,
                rot_error_type="axis_angle"
            )
            pose_err = torch.cat([p_err, r_err], dim=1)  # N×6
            delta_q  = self._compute_delta_joint_pos(pose_err, jacobian)

        return joint_pos + delta_q

    def _compute_delta_joint_pos(
        self,
        delta_pose: torch.Tensor,
        jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        delta_pose: (N,D)  D=3 or 6
        jacobian:   (N,D,num_joints)
        returns delta_q: (N,num_joints)
        """
        N, D, num_j = jacobian.shape
        if self.ik_method == "pinv":
            k = self.ik_params.get("k_val", 1.0)
            J_pinv = torch.linalg.pinv(jacobian)
            dq = k * (J_pinv @ delta_pose.unsqueeze(-1)).squeeze(-1)

        elif self.ik_method == "svd":
            k = self.ik_params.get("k_val", 1.0)
            min_s = self.ik_params.get("min_singular_value", 0.0)
            dq = []
            for i in range(N):
                J = jacobian[i]                     # D×nj
                U, S, Vh = torch.linalg.svd(J)
                S_inv = torch.where(S > min_s, 1.0/S, torch.zeros_like(S))
                J_pinv = (Vh.T[:, :D] @ torch.diag(S_inv) @ U.T)
                dq.append(k * (J_pinv @ delta_pose[i]))
            dq = torch.stack(dq, dim=0)

        elif self.ik_method == "trans":
            k = self.ik_params.get("k_val", 1.0)
            Jt = jacobian.transpose(1,2)            # N×nj×D
            dq = k * (Jt @ delta_pose.unsqueeze(-1)).squeeze(-1)

        elif self.ik_method == "dls":
            lam = self.ik_params.get("lambda_val", 0.01)
            # (N×D×D)
            JJT = jacobian @ jacobian.transpose(1,2)
            lamI = (lam**2) * torch.eye(D, device=jacobian.device).unsqueeze(0).expand(N, D, D)
            inv = torch.linalg.inv(JJT + lamI)
            Jt  = jacobian.transpose(1,2)           # N×nj×D
            dq = (Jt @ (inv @ delta_pose.unsqueeze(-1))).squeeze(-1)

        else:
            raise ValueError(f"Unknown ik_method '{self.ik_method}'")

        return dq
