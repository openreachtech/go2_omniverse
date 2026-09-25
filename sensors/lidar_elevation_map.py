"""Height grid built from a LiDAR fan, ported from unitree_rl_lab
(unitree_rl_lab/tasks/locomotion/mdp/lidar_elevation_map.py, origin/feat/mid360 branch)
for running its go2_height_map (Go2-Perceptive-Mid360-Phase5-C) exported policy inside
this project.

Ported near-verbatim. The one change: upstream's ``exclude_half_extent_x/y > 0`` branch
called ``_height_scan_indices`` from a sibling ``observations.py`` module (a much larger
file, not needed here) to crop the grid to a body-exclusion rectangle. This project's own
go2_height_map run always passes ``exclude_half_extent_x/y = -1.0`` (keep every cell — see
velocity_env_cfg_mid360.py's ``_mid360_map_term``), so that branch is dead code for us and
is left as a ``NotImplementedError`` instead of chasing the extra dependency.

See the class docstring below (also ported verbatim) for what this term actually does:
bins a LiDAR fan's returns into a body-centered height grid with motion-compensated
holds, matching the observation the policy was trained on. Noise (``LidarNoiseCfg``) is
for training-time domain randomization; this project's inference always passes
``noise=None`` for a clean run.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_angle_axis

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_UNOBSERVED = 1.0e4
"""Sentinel for a cell no beam reached. Large, so it loses every ``amin``."""


@configclass
class LidarNoiseConditionCfg:
    """One noise condition: how badly the sensor behaves for this episode.

    Standard deviations, in metres and degrees. Everything scales with
    ``LidarNoiseCfg.scale``, so tuning overall severity does not mean editing these.
    """

    probability: float = 0.0
    """Relative chance of an episode drawing this condition. Normalised across conditions."""
    range_std: float = 0.0
    """Per-ray distance error (m), resampled every step."""
    tilt_step_std: float = 0.0
    """Per-step tilt of the returns about the sensor origin (deg)."""
    tilt_episode_std: float = 0.0
    """Per-episode tilt held for the run (deg)."""
    outlier_prob: float = 0.0
    """Per-ray, per-step chance of a grossly wrong distance."""
    outlier_range: float = 0.0
    """Magnitude of an outlier's distance error (m), uniform in +-this."""

    odom_xy_step_std: float = 0.0
    """Per-step odometry translation error, as a fraction of the distance moved that step."""
    odom_xy_bias_std: float = 0.0
    """Per-episode odometry translation error, same units. Held for the run."""
    odom_yaw_step_std: float = 0.0
    """Per-step odometry heading error, in degrees per metre travelled."""
    odom_yaw_bias_std: float = 0.0
    """Per-episode odometry heading drift, in degrees per metre travelled. Held for the run."""


@configclass
class LidarNoiseCfg:
    """Per-episode noise conditions, plus one knob to scale all of them.

    Not used by this project's inference path (always constructed with ``noise=None``
    on the observation call) -- kept so the ported ``LidarElevationMap`` matches upstream
    exactly and can have noise re-enabled later without further porting.
    """

    weak: LidarNoiseConditionCfg = LidarNoiseConditionCfg(
        probability=0.60, range_std=0.01, tilt_step_std=0.5, tilt_episode_std=0.25,
        outlier_prob=0.005, outlier_range=0.15,
        odom_xy_step_std=0.01, odom_xy_bias_std=0.01,
        odom_yaw_step_std=0.5, odom_yaw_bias_std=0.5,
    )
    nominal: LidarNoiseConditionCfg = LidarNoiseConditionCfg(
        probability=0.30, range_std=0.02, tilt_step_std=1.0, tilt_episode_std=0.5,
        outlier_prob=0.01, outlier_range=0.30,
        odom_xy_step_std=0.02, odom_xy_bias_std=0.02,
        odom_yaw_step_std=1.0, odom_yaw_bias_std=1.0,
    )
    strong: LidarNoiseConditionCfg = LidarNoiseConditionCfg(
        probability=0.10, range_std=0.04, tilt_step_std=2.0, tilt_episode_std=1.0,
        outlier_prob=0.03, outlier_range=0.60,
        odom_xy_step_std=0.05, odom_xy_bias_std=0.05,
        odom_yaw_step_std=3.0, odom_yaw_bias_std=3.0,
    )

    scale: float = 1.0
    """Multiplies every magnitude above. 0.0 disables noise entirely."""

    num_steps_per_env: int = 24
    """Rollout length of one iteration, to turn env steps into iterations for the ramp."""
    start_iteration: int = 0
    """Iterations of noise-free returns before the ramp begins."""
    full_iteration: int = 0
    """Iteration at which ``scale`` is reached. Equal to ``start_iteration`` means no ramp."""


def _curriculum_level(common_step_counter: int, cfg: LidarNoiseCfg) -> float:
    """Fraction of the configured magnitude to apply at the current iteration."""
    if cfg.full_iteration <= cfg.start_iteration:
        return 1.0
    iteration = common_step_counter // max(cfg.num_steps_per_env, 1)
    if iteration >= cfg.full_iteration:
        return 1.0
    if iteration <= cfg.start_iteration:
        return 0.0
    return (iteration - cfg.start_iteration) / (cfg.full_iteration - cfg.start_iteration)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw angle of a (w, x, y, z) quaternion. Shape (..., 4) -> (...,)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class LidarElevationMap(ManagerTermBase):
    """Bin a LiDAR fan's returns into the body-centered height grid.

    The heavy state is one buffer of held cell values; everything else is a
    stateless reduction over the fan's hits. See ``_advance_hold`` for the motion
    compensation that carries a held value with the robot's motion between steps, so a
    value describes the same patch of ground it was measured on rather than riding along
    with the body.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._resolution: float = p["resolution"]
        self._size: tuple[float, float] = p["size"]
        self._offset: float = p["offset"]
        self._flat_fill: float = p["flat_fill"]
        self._lidar_offset: tuple[float, float, float] = p["lidar_offset"]

        self._num_x = round(self._size[0] / self._resolution) + 1
        self._num_y = round(self._size[1] / self._resolution) + 1
        self._num_cells = self._num_x * self._num_y
        self._x0 = -self._size[0] / 2 + p["scanner_offset_xy"][0]
        self._y0 = -self._size[1] / 2 + p["scanner_offset_xy"][1]

        # ordering="yx" (idx = ix * num_y + iy).
        # Non-positive extents mean keep every cell: a sensor mounted low enough to see
        # under the trunk has no blind rectangle to cut out, and passing 0.0 would still
        # drop the single cell sitting exactly on the body origin.
        if p["exclude_half_extent_x"] <= 0.0 and p["exclude_half_extent_y"] <= 0.0:
            self._keep_indices = torch.arange(self._num_cells, device=self.device)
        else:
            # Upstream crops to a body-exclusion rectangle here via a helper from a
            # sibling observations.py; not ported (see module docstring) since this
            # project's own go2_height_map usage never takes this branch.
            raise NotImplementedError(
                "exclude_half_extent_x/y > 0 is not supported by this port — "
                "go2_omniverse's mid360 setup always passes -1.0 (keep every cell)."
            )

        # Cell centers in the yaw-aligned base frame, for the diagnostics and markers.
        cx = torch.linspace(self._x0, self._x0 + self._size[0], self._num_x, device=self.device)
        cy = torch.linspace(self._y0, self._y0 + self._size[1], self._num_y, device=self.device)
        gx, gy = torch.meshgrid(cx, cy, indexing="ij")
        self._cell_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (num_cells, 2)

        kept_xy = self._cell_xy.index_select(0, self._keep_indices)
        # Cells outside the fan's azimuth wedge can never receive a beam, so counting
        # them as unobserved would report a field-of-view choice as a density problem.
        # With a full turn every cell qualifies and the mask is all-true.
        h_fov = p["horizontal_fov"]
        if h_fov[1] - h_fov[0] >= 359.9:
            self._in_fov = torch.ones(kept_xy.shape[0], dtype=torch.bool, device=self.device)
        else:
            azimuth = torch.rad2deg(
                torch.atan2(kept_xy[:, 1] - self._lidar_offset[1], kept_xy[:, 0] - self._lidar_offset[0])
            )
            self._in_fov = (azimuth >= h_fov[0]) & (azimuth <= h_fov[1])
        radius = torch.linalg.vector_norm(kept_xy - kept_xy.new_tensor(self._lidar_offset[:2]), dim=-1)
        self._band_near = self._in_fov & (radius < 0.30)
        self._band_mid = self._in_fov & (radius >= 0.30) & (radius < 0.50)
        self._band_far = self._in_fov & (radius >= 0.50)

        self._hold = torch.full((self.num_envs, self._num_cells), self._flat_fill, device=self.device)

        # Motion compensation. The held map is indexed by cell *relative to the robot*, so
        # without this a value stays in the same slot while the ground under that slot slides
        # away -- a wall height measured 40 cm ahead is still sitting 40 cm ahead once the
        # robot has walked past the wall. See _advance_hold.
        self._motion_compensation: bool = bool(p.get("motion_compensation", True))
        self._prev_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_yaw = torch.zeros(self.num_envs, device=self.device)
        self._prev_z = torch.zeros(self.num_envs, device=self.device)
        # Which cells hold a real measurement rather than the flat fill.
        self._measured = torch.zeros(
            self.num_envs, self._num_cells, dtype=torch.bool, device=self.device
        )
        # False until this env has a pose to measure displacement against: the first step of
        # an episode, and the first step after a reset teleports the robot.
        self._pose_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Noise conditions, drawn per episode. Absent config means a noise-free sensor.
        self._noise: LidarNoiseCfg | None = p.get("noise")
        if self._noise is not None:
            conds = [self._noise.weak, self._noise.nominal, self._noise.strong]

            def _column(attr: str) -> torch.Tensor:
                return torch.tensor([getattr(c, attr) for c in conds], device=self.device)

            weights = _column("probability")
            if torch.any(weights < 0.0) or weights.sum() <= 0.0:
                raise ValueError("LiDAR noise condition probabilities must be >= 0 and sum to > 0.")
            self._cond_weights = weights / weights.sum()
            self._range_std = _column("range_std")
            self._tilt_step_std = torch.deg2rad(_column("tilt_step_std"))
            self._tilt_episode_std = torch.deg2rad(_column("tilt_episode_std"))
            self._outlier_prob = _column("outlier_prob")
            self._outlier_range = _column("outlier_range")
            self._odom_xy_step_std = _column("odom_xy_step_std")
            self._odom_xy_bias_std = _column("odom_xy_bias_std")
            self._odom_yaw_step_std = torch.deg2rad(_column("odom_yaw_step_std"))
            self._odom_yaw_bias_std = torch.deg2rad(_column("odom_yaw_bias_std"))
            self._odom_bias_xy = torch.zeros(self.num_envs, 2, device=self.device)
            self._odom_bias_yaw = torch.zeros(self.num_envs, device=self.device)
            self._condition = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._episode_tilt = torch.zeros(self.num_envs, 2, device=self.device)
            self._draw_conditions(torch.arange(self.num_envs, device=self.device))

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._hold[ids] = self._flat_fill
        self._measured[ids] = False
        # A reset moves the robot somewhere else entirely; carrying the pre-reset pose into
        # the displacement would shift the fresh map by the length of the teleport.
        self._pose_valid[ids] = False
        if self._noise is not None:
            self._draw_conditions(ids)

    def _draw_conditions(self, env_ids: torch.Tensor) -> None:
        """Pick a noise condition per episode, and the tilt this run is stuck with."""
        if env_ids.numel() == 0:
            return
        drawn = torch.multinomial(self._cond_weights, env_ids.numel(), replacement=True)
        self._condition[env_ids] = drawn
        self._episode_tilt[env_ids] = (
            torch.randn(env_ids.numel(), 2, device=self.device) * self._tilt_episode_std[drawn].unsqueeze(-1)
        )
        self._odom_bias_xy[env_ids] = (
            torch.randn(env_ids.numel(), 2, device=self.device) * self._odom_xy_bias_std[drawn].unsqueeze(-1)
        )
        self._odom_bias_yaw[env_ids] = (
            torch.randn(env_ids.numel(), device=self.device) * self._odom_yaw_bias_std[drawn]
        )

    def _corrupt_odometry(
        self, step_xy: torch.Tensor, step_yaw: torch.Tensor, level: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace a step's true ego-motion with what an estimator would have reported."""
        travelled = torch.linalg.vector_norm(step_xy, dim=-1, keepdim=True)
        cond = self._condition
        xy_error = self._odom_bias_xy + torch.randn_like(step_xy) * (
            level * self._odom_xy_step_std[cond]
        ).unsqueeze(-1)
        yaw_error = self._odom_bias_yaw + torch.randn_like(step_yaw) * (
            level * self._odom_yaw_step_std[cond]
        )
        return (
            step_xy + travelled * xy_error * level,
            step_yaw + travelled.squeeze(-1) * yaw_error * level,
        )

    def _advance_hold(
        self, root_xy: torch.Tensor, root_z: torch.Tensor, yaw: torch.Tensor, level: float = 0.0
    ) -> None:
        """Carry the held map with the robot, so a held value stays on its patch of ground."""
        if not self._motion_compensation:
            return
        if bool(self._pose_valid.any()):
            cos_prev, sin_prev = torch.cos(self._prev_yaw), torch.sin(self._prev_yaw)
            delta = root_xy - self._prev_xy
            # R(-yaw_prev) @ (xy_now - xy_prev): the translation seen from the old body frame.
            ux = cos_prev * delta[:, 0] + sin_prev * delta[:, 1]
            uy = -sin_prev * delta[:, 0] + cos_prev * delta[:, 1]
            step_xy = torch.stack([ux, uy], dim=-1)
            step_yaw = yaw - self._prev_yaw
            if self._noise is not None and level > 0.0:
                step_xy, step_yaw = self._corrupt_odometry(step_xy, step_yaw, level)
            ux, uy = step_xy[:, 0], step_xy[:, 1]
            cos_d = torch.cos(step_yaw).unsqueeze(-1)
            sin_d = torch.sin(step_yaw).unsqueeze(-1)

            cell_x, cell_y = self._cell_xy[:, 0].unsqueeze(0), self._cell_xy[:, 1].unsqueeze(0)
            qx = cos_d * cell_x - sin_d * cell_y + ux.unsqueeze(-1)
            qy = sin_d * cell_x + cos_d * cell_y + uy.unsqueeze(-1)

            # Fractional source indices, then grid_sample's [-1, 1] with align_corners=True.
            src_x = (qx - self._x0) / self._resolution
            src_y = (qy - self._y0) / self._resolution
            norm_x = 2.0 * src_x / (self._num_x - 1) - 1.0
            norm_y = 2.0 * src_y / (self._num_y - 1) - 1.0
            # _hold is (N, num_x * num_y) with x as the outer axis, so as an image it is
            # H = num_x, W = num_y -- and grid_sample's last axis indexes W first.
            grid = torch.stack([norm_y, norm_x], dim=-1).view(
                self.num_envs, self._num_x, self._num_y, 2
            )
            source = torch.stack(
                [self._hold, self._measured.to(self._hold.dtype), torch.ones_like(self._hold)],
                dim=1,
            ).view(self.num_envs, 3, self._num_x, self._num_y)
            sampled = torch.nn.functional.grid_sample(
                source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(self.num_envs, 3, self._num_cells)

            # Channel 2 is all ones inside the old window, so anything short of 1 means the
            # sample reached past its edge; channel 1 says whether what it found was a real
            # measurement or the flat fill.
            inside = sampled[:, 2] > 0.999
            measured = inside & (sampled[:, 1] > 0.5)
            shifted = torch.where(
                measured,
                sampled[:, 0] + (root_z - self._prev_z).unsqueeze(-1),
                torch.full_like(sampled[:, 0], self._flat_fill),
            )
            keep = self._pose_valid.unsqueeze(-1)
            self._hold = torch.where(keep, shifted, self._hold)
            self._measured = torch.where(keep, measured, self._measured)

        self._prev_xy = root_xy.clone()
        self._prev_yaw = yaw.clone()
        self._prev_z = root_z.clone()
        self._pose_valid = torch.ones_like(self._pose_valid)

    def _perturb(
        self, rel: torch.Tensor, finite: torch.Tensor, level: float, min_range: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the paper's Position, Outliers and Tilt to the returns, per ray."""
        cond = self._condition
        distance = rel.norm(dim=-1, keepdim=True)
        distance = torch.where(finite.unsqueeze(-1), distance, torch.zeros_like(distance))
        direction = rel / distance.clamp(min=1.0e-6)

        error = torch.randn_like(distance) * (level * self._range_std[cond]).view(-1, 1, 1)
        outlier_mag = (level * self._outlier_range[cond]).view(-1, 1, 1)
        is_outlier = torch.rand_like(distance) < self._outlier_prob[cond].view(-1, 1, 1)
        outlier = (torch.rand_like(distance) * 2.0 - 1.0) * outlier_mag
        error = torch.where(is_outlier, outlier, error)
        new_distance = distance + error
        if min_range > 0.0:
            finite = finite & (new_distance.squeeze(-1) >= min_range)
        rel = direction * new_distance.clamp(min=0.0)

        step_tilt = torch.randn(self.num_envs, 2, device=self.device) * self._tilt_step_std[cond].unsqueeze(-1)
        tilt = level * (step_tilt + self._episode_tilt)
        angle = torch.linalg.vector_norm(tilt, dim=-1)
        axis = torch.nn.functional.normalize(
            torch.stack([tilt[:, 0], tilt[:, 1], torch.zeros_like(angle)], dim=-1), dim=-1, eps=1.0e-9
        )
        rot = matrix_from_quat(quat_from_angle_axis(angle, axis))
        return torch.einsum("nij,nrj->nri", rot, rel), finite

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        offset: float = 0.0,
        resolution: float = 0.05,
        size: tuple[float, float] = (1.4, 1.0),
        scanner_offset_xy: tuple[float, float] = (0.0, 0.0),
        exclude_half_extent_x: float = 0.30,
        exclude_half_extent_y: float = 0.20,
        lidar_offset: tuple[float, float, float] = (0.19, 0.0, 0.10),
        horizontal_fov: tuple[float, float] = (-180.0, 180.0),
        flat_fill: float = 0.0,
        noise: LidarNoiseCfg | None = None,
        min_range: float = 0.0,
        motion_compensation: bool = True,
        debug_vis: bool = False,
        debug_vis_env_index: int | None = 0,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        asset = env.scene[asset_cfg.name]
        hits_w = sensor.data.ray_hits_w  # (N, num_rays, 3)
        root_pos = asset.data.root_pos_w  # (N, 3)
        finite = torch.isfinite(hits_w).all(dim=-1)

        level = 0.0
        if self._noise is not None:
            level = self._noise.scale * _curriculum_level(env.common_step_counter, self._noise)
            env.lidar_noise_level = level
            if level > 0.0:
                origin = sensor.data.pos_w + quat_apply(sensor.data.quat_w, sensor.ray_starts[:, 0])
                perturbed, finite = self._perturb(
                    hits_w - origin.unsqueeze(1), finite, level, min_range
                )
                hits_w = origin.unsqueeze(1) + perturbed

        # Carry the held map onto this step's pose before anything new is merged into it,
        # so both are describing the same ground.
        yaw_now = _yaw_from_quat(asset.data.root_quat_w)
        self._advance_hold(root_pos[:, 0:2], root_pos[:, 2], yaw_now, level)

        rel = hits_w - root_pos.unsqueeze(1)
        yaw = yaw_now.unsqueeze(1)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        bx = cos_y * rel[..., 0] + sin_y * rel[..., 1]
        by = -sin_y * rel[..., 0] + cos_y * rel[..., 1]

        ix = torch.round((bx - self._x0) / self._resolution).long()
        iy = torch.round((by - self._y0) / self._resolution).long()
        valid = (
            (ix >= 0)
            & (ix < self._num_x)
            & (iy >= 0)
            & (iy < self._num_y)
            & finite
        )

        heights = root_pos[:, 2:3] - hits_w[..., 2] - offset
        heights = torch.where(valid, heights, heights.new_full((), _UNOBSERVED))
        flat_idx = torch.where(valid, ix * self._num_y + iy, torch.zeros_like(ix))

        grid = torch.full_like(self._hold, _UNOBSERVED)
        grid.scatter_reduce_(1, flat_idx, heights, reduce="amin", include_self=True)

        unobserved = grid >= _UNOBSERVED * 0.5
        grid = torch.where(unobserved, self._hold, grid)
        self._hold = grid
        self._measured |= ~unobserved

        kept = grid.index_select(1, self._keep_indices)
        env.lidar_map_unobserved_cells = unobserved
        self._record_diagnostics(env, unobserved.index_select(1, self._keep_indices))

        if debug_vis:
            self._visualize(env, asset, kept, unobserved, offset, debug_vis_env_index)
        return kept

    def _record_diagnostics(self, env: ManagerBasedRLEnv, unobserved_kept: torch.Tensor) -> None:
        def rate(mask: torch.Tensor) -> float:
            if not bool(mask.any()):
                return 0.0
            return float(unobserved_kept[:, mask].float().mean())

        env.lidar_map_unobserved_rate = rate(self._in_fov)
        env.lidar_map_unobserved_near = rate(self._band_near)
        env.lidar_map_unobserved_mid = rate(self._band_mid)
        env.lidar_map_unobserved_far = rate(self._band_far)

    def _visualize(
        self,
        env: ManagerBasedRLEnv,
        asset,
        kept: torch.Tensor,
        unobserved: torch.Tensor,
        offset: float,
        env_index: int | None,
    ) -> None:
        """Green = measured this step, red = held from an earlier step."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if not hasattr(self, "_visualizer"):
            self._visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Go2LidarMap",
                    markers={
                        "measured": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        ),
                        "held": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    },
                )
            )

        kept_xy = self._cell_xy.index_select(0, self._keep_indices)  # (K, 2)
        held = unobserved.index_select(1, self._keep_indices)
        if env_index is None:
            env_ids = torch.arange(kept.shape[0], device=self.device)
        else:
            env_ids = torch.tensor([min(max(env_index, 0), kept.shape[0] - 1)], device=self.device)

        root_pos = asset.data.root_pos_w[env_ids]
        yaw = _yaw_from_quat(asset.data.root_quat_w[env_ids]).unsqueeze(-1)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cell_x, cell_y = kept_xy[:, 0].unsqueeze(0), kept_xy[:, 1].unsqueeze(0)

        shown = kept[env_ids]
        if self.cfg.clip is not None:
            shown = shown.clamp(min=self.cfg.clip[0], max=self.cfg.clip[1])
        positions = torch.stack(
            [
                root_pos[:, 0:1] + cos_y * cell_x - sin_y * cell_y,
                root_pos[:, 1:2] + sin_y * cell_x + cos_y * cell_y,
                root_pos[:, 2:3] - shown - offset,
            ],
            dim=-1,
        ).reshape(-1, 3)
        marker_indices = held[env_ids].reshape(-1).long()

        finite = torch.isfinite(positions).all(dim=-1)
        if not bool(finite.any()):
            return
        self._visualizer.visualize(
            translations=positions[finite], marker_indices=marker_indices[finite]
        )
