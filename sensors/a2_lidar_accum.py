"""World-fixed accumulation height map for A2's front/rear LiDAR pair.

Ported (trimmed to a single continuous run, not an RL training episode) from the
reference play script that anaguma_perceptive_mid360_phase2... no -- from the A2
distilled-policy reference script (robots/a2/tmp/a2_share/code/a2_package/lidar_accum.py,
function ``lidar_accum_scan``), which is what tb01_a2_distill_distill2_20500 was verified
against. Unlike Go2/Anaguma's body-relative ``LidarElevationMap`` (a bounded grid,
re-centred on the robot every step), this is a **world-fixed** grid that both LiDARs
write into over the whole run, from which a body-relative, yaw-rotated 17x11 window is
read out every step.

Dropped from the original (training-only concerns, irrelevant to a single continuous
--policy_path run): per-episode reset/clearing (we just accumulate from t=0), the
FOLLOW mode, the per-episode/per-step localization-noise bias (dxy/dyaw — domain
randomization, same reasoning as go2_height_map's LidarElevationMap noise=None), the
multi-env state dict, and all of the A2_* debugging env-var switches.

Kept: the core world grid (amax-per-cell accumulation), the exact window-extraction
geometry/formula, and the final uniform +-0.1 observation noise -- the reference script's
own comment states this noise is *not* a training-time-only artifact to drop at
inference: it's present in the ObsTerm itself (``enable_corruption=True``) and omitting
it was found to visibly change behavior, so it stays on by default here.
"""

from __future__ import annotations

import math

import torch
from isaaclab.sensors import RayCasterCfg, patterns

# --- Mount (a2.urdf, base_link frame) --------------------------------------------
FRONT_POS = (0.33767, 0.0, 0.08134)
FRONT_ROT = (0.5, 0.5, 0.5, 0.5)            # (w, x, y, z)
REAR_POS = (-0.27997, 0.0, 0.08734)
REAR_ROT = (0.5, 0.5, -0.5, -0.5)

# --- Rays --------------------------------------------------------------------------
# Vertical FOV is a Mid-360 stand-in (A2's real LiDAR spec is unconfirmed) carried over
# unchanged from the reference script.
VERT_FOV = (-7.0, 52.0)
HORIZ_FOV = (-180.0, 180.0)
RAY_STEP_DEG = 3.0
R_MIN, R_MAX = 0.15, 6.0

# --- Output window (base-centred) ---------------------------------------------------
WIN_X = (-0.8, 0.8)
WIN_Y = (-0.5, 0.5)
RES = 0.1
NX = int(round((WIN_X[1] - WIN_X[0]) / RES)) + 1     # 17
NY = int(round((WIN_Y[1] - WIN_Y[0]) / RES)) + 1     # 11

# --- World-fixed grid -----------------------------------------------------------
HALF = 6.0
GN = int(round(2 * HALF / RES)) + 1                   # 121
GRID_CELLS = GN * GN

# Phase offset so neither window axis' cell centres land exactly on a grid boundary
# (see the reference script's comment: 0.25 is the only value under which all 187
# window cells land in distinct grid cells for both the 1.6 m x- and 1.0 m y-window).
PHASE = 0.25


def channels_for(step_deg: float = RAY_STEP_DEG) -> int:
    return int(round((VERT_FOV[1] - VERT_FOV[0]) / step_deg)) + 1


def lidar_cfg(which: str, prim_path: str, mesh_prim_paths: list[str]) -> RayCasterCfg:
    """Front/rear RayCasterCfg. mesh_prim_paths must be a single-element list (IsaacLab's
    RayCaster only accepts one mesh prim — same constraint height_scanner_env/
    mid360_scanner work around elsewhere in this repo)."""
    pos, rot = (FRONT_POS, FRONT_ROT) if which == "front" else (REAR_POS, REAR_ROT)
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=pos, rot=rot),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=channels_for(),
            vertical_fov_range=VERT_FOV,
            horizontal_fov_range=HORIZ_FOV,
            horizontal_res=RAY_STEP_DEG,
        ),
        max_distance=R_MAX,
        debug_vis=False,
        mesh_prim_paths=mesh_prim_paths,
    )


class A2LidarAccumulator:
    """Stateful world-fixed height accumulator for a single A2 robot (env 0 only).

    Call once per sim step after the front/rear RayCaster sensors have updated; returns
    a (1, 374) tensor: 187 heights + 187 unseen-flags, matching
    anaguma_perceptive_mid360_phase2... (A2's) training observation exactly.
    """

    def __init__(
        self, env, device, offset: float = 0.0, scale: float = 1.0 / 5.0, noise: float = 0.1,
        front_sensor: str = "lidar_front", rear_sensor: str = "lidar_rear",
    ):
        self._env = env
        self._device = device
        self._offset = offset
        self._scale = scale
        self._noise = noise
        self._front_name = front_sensor
        self._rear_name = rear_sensor
        self._grid = torch.full((GRID_CELLS,), float("nan"), device=device, dtype=torch.float32)
        robot = env.scene["robot"]
        self._origin = robot.data.root_pos_w[0, :2].clone().to(torch.float32)

        ix = torch.arange(NX, device=device, dtype=torch.float32) * RES + WIN_X[0]
        iy = torch.arange(NY, device=device, dtype=torch.float32) * RES + WIN_Y[0]
        gyv, gxv = torch.meshgrid(iy, ix, indexing="ij")
        self._lx = gxv.reshape(-1)   # (187,) local x offsets of the window cells
        self._ly = gyv.reshape(-1)   # (187,) local y offsets

    def _cell(self, v: torch.Tensor, origin: float) -> torch.Tensor:
        return torch.floor((v - origin + HALF) / RES + PHASE).long()

    def __call__(self) -> torch.Tensor:
        sensors = self._env.scene.sensors
        front_hits = sensors[self._front_name].data.ray_hits_w[0]   # (R, 3)
        rear_hits = sensors[self._rear_name].data.ray_hits_w[0]
        front_origin = sensors[self._front_name].data.pos_w[0]
        rear_origin = sensors[self._rear_name].data.pos_w[0]
        hits = torch.cat([front_hits, rear_hits], dim=0)
        rng = torch.cat([
            torch.linalg.norm(front_hits - front_origin, dim=-1),
            torch.linalg.norm(rear_hits - rear_origin, dim=-1),
        ], dim=0)

        robot = self._env.scene["robot"]
        base_pos = robot.data.root_pos_w[0].to(torch.float32)
        yaw = _yaw_of(robot.data.root_quat_w[0])

        wx, wy, wz = hits[:, 0], hits[:, 1], hits[:, 2]
        ok = torch.isfinite(wz) & (rng >= R_MIN) & (rng <= R_MAX)

        gx = self._cell(wx, float(self._origin[0]))
        gy = self._cell(wy, float(self._origin[1]))
        in_grid = (gx >= 0) & (gx < GN) & (gy >= 0) & (gy < GN)
        ok &= in_grid
        idx = (gy.clamp(0, GN - 1) * GN + gx.clamp(0, GN - 1))
        idx = torch.where(ok, idx, torch.full_like(idx, GRID_CELLS))

        # Highest hit wins per cell this step (so a thin raised obstacle isn't lost to a
        # floor hit landing in the same cell).
        acc = torch.full((GRID_CELLS + 1,), float("-inf"), device=self._device, dtype=torch.float32)
        acc.scatter_reduce_(0, idx, torch.where(ok, wz, torch.full_like(wz, float("-inf"))),
                             reduce="amax", include_self=True)
        new = acc[:GRID_CELLS]
        wrote = torch.isfinite(new)
        self._grid = torch.where(wrote, new, self._grid)

        # Base-centred, yaw-rotated 17x11 window into the world-fixed grid.
        cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
        qx = base_pos[0] + cy * self._lx - sy * self._ly
        qy = base_pos[1] + sy * self._lx + cy * self._ly
        cx = self._cell(qx, float(self._origin[0])).clamp(0, GN - 1)
        cyi = self._cell(qy, float(self._origin[1])).clamp(0, GN - 1)
        cell = cyi * GN + cx
        z = self._grid[cell]

        unseen = ~torch.isfinite(z)
        h = (base_pos[2] - z - self._offset) * self._scale
        h = torch.where(unseen, torch.zeros_like(h), h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.cat([h, unseen.to(torch.float32)], dim=0).unsqueeze(0)  # (1, 374)
        if self._noise > 0.0:
            out = out + (torch.rand_like(out) * 2.0 - 1.0) * self._noise
        return out


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
