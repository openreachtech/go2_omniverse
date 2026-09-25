# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
import os
import torch
from dataclasses import MISSING
from typing import Literal
import argparse  


from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCaster, RayCasterCfg, patterns

from sensors import LivoxPatternCfg, RollingLivoxSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets import UNITREE_GO2_CFG
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from terrain_cfg import ROUGH_TERRAINS_CFG
from robots.g1.config import G1_CFG
from robots.anaguma.config import ANAGUMA_CFG

from omniverse_sim import args_cli


# ponytail: resolve Isaac's bundled studio HDRI once at module load (a plain string, so
# it survives @configclass deepcopy — module objects in a class body cannot be pickled).
# Glob because the extscache path is version-pinned and shifts on Isaac upgrades; empty
# string falls back to a flat neutral dome if the asset ever moves.
def _find_studio_hdri():
    import glob, os
    import isaacsim
    hits = glob.glob(
        os.path.dirname(isaacsim.__file__) + "/**/domeLight/photo_studio_01_4k.hdr",
        recursive=True,
    )
    return hits[0] if hits else ""


_STUDIO_HDRI = _find_studio_hdri()


base_command = {}


def constant_commands(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    global base_command
    """The generated command from the command generator."""
    tensor_lst = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    for i in range(env.num_envs):
        tensor_lst[i] = torch.tensor(base_command[str(i)], dtype=torch.float32, device=env.device)
    return tensor_lst


class RayCasterSafeVis(RayCaster):
    """RayCaster whose debug_vis marker draw tolerates an all-miss scan.

    IsaacLab's own RayCaster._debug_vis_callback() calls visualize() with zero points
    whenever every ray misses (e.g. height_scanner_env when the robot isn't currently
    standing over the custom env's mesh at all) and VisualizationMarkers.visualize()
    raises ValueError on an empty input — crashing the sim every frame that happens.
    """

    def _debug_vis_callback(self, event):
        if self._data.ray_hits_w is None:
            return
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points.numel() == 0:
            return
        self.ray_visualizer.visualize(viz_points)


def height_scan_merged(
    env: ManagerBasedRLEnvCfg, sensor_cfg: SceneEntityCfg, sensor_cfg_2: SceneEntityCfg, offset: float = 0.5
) -> torch.Tensor:
    """Like mdp.height_scan, but merges two RayCasters (RayCaster itself only accepts one
    mesh_prim_path) by keeping whichever surface each ray hits first — the higher one, i.e.
    the smaller sensor-to-hit distance. A ray that misses a mesh entirely (e.g. the custom
    env has no geometry under that point) gets ray_hits_w == +inf there (IsaacLab's documented
    miss convention, see raycast_mesh() in isaaclab/utils/warp/ops.py) — sensor_z - inf is
    -inf, which would otherwise always "win" the min() as if it were the closest possible
    surface, so misses are pinned to +inf distance before merging.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_2 = env.scene.sensors[sensor_cfg_2.name]
    dist = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]
    hit_z_2 = sensor_2.data.ray_hits_w[..., 2]
    dist_2 = sensor_2.data.pos_w[:, 2].unsqueeze(1) - hit_z_2
    dist_2 = torch.where(torch.isinf(hit_z_2), torch.full_like(dist_2, float("inf")), dist_2)
    return torch.min(dist, dist_2) - offset


def _build_mid360_scanner_cfg() -> RollingLivoxSensorCfg | None:
    """RollingLivoxSensorCfg for --lidar_map, or None when the flag is off.

    A module-level function rather than inline in MySceneCfg's class body: any name
    assigned directly in an @configclass body becomes a dataclass field, and
    InteractiveScene misreads a plain str/float local (not a SceneEntityCfg-like object)
    as a scene entity to instantiate and raises ValueError — the same trap
    _HEIGHT_SCAN_MESH_PATHS hit earlier in this file's history.

    Mount: the real L1 (utlidar) pose from go2_description.urdf's radar_joint,
    xyz="0.28945 0 -0.046825" rpy="0 2.8782 0" -- nose tip, pitched 164.9 deg so the
    sensor hangs nearly upside down looking out and down through the nose aperture.
    Ported from unitree_rl_lab's velocity_env_cfg_mid360.py (GO2_L1_MOUNT / GO2_L1_ROT),
    which derives the same values from that URDF. mesh_prim_paths targets the same single
    mesh as height_scanner/height_scanner_env (RayCaster only accepts one), so all three
    agree on what "the ground" is in a custom env.
    """
    if not args_cli.lidar_map:
        return None
    mesh_target = (
        f"/World/{args_cli.custom_env}_scan"
        if (
            args_cli.terrain == "flat"
            and args_cli.height_scan == "mesh"
            and os.path.isfile(f"./envs/{args_cli.custom_env}.usd")
        )
        else "/World/ground"
    )
    l1_pitch = 2.8782  # rad
    return RollingLivoxSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RollingLivoxSensorCfg.OffsetCfg(
            pos=(0.28945, 0.0, -0.046825),
            rot=(math.cos(l1_pitch / 2), 0.0, math.sin(l1_pitch / 2), 0.0),
        ),
        ray_alignment="base",
        pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=4000, downsample=2),
        mesh_prim_paths=[mesh_target],
        max_distance=20.0,
        min_range=0.2,
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        enable_sensor_noise=False,
        update_frequency=50.0,
        debug_vis=False,
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    if args_cli.terrain == "flat":
    
        # flat terrain
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            debug_vis=False,
        )
    else:
        terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        ),
        debug_vis=False,
        )

    # robots
    robot: ArticulationCfg = MISSING

    # debug_vis stays off on both sensors below — each RayCaster draws its own RAW hits
    # independently, so with two sensors that doubles up into two overlapping marker grids
    # instead of one that matches what the policy actually sees. run_sim() draws the
    # merged result (same logic as height_scan_merged()) with a single marker set instead.
    height_scanner = RayCasterCfg(
        class_type=RayCasterSafeVis,
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # RayCaster only accepts one mesh prim path, so a custom env's own geometry needs a
    # second sensor rather than being added to height_scanner above — merged back together
    # in height_scan_merged(). Points at "..._scan", a single combined Mesh prim that
    # setup_custom_env() builds from every Mesh under the custom env (RayCaster itself only
    # reads the first Mesh prim it finds, which misses multi-mesh scenes like ramps/steps
    # authored as separate cubes). None (disabled) unless a custom env is actually loaded
    # AND --height_scan mesh (the default) — --height_scan ground forces ground-only.
    if (
        args_cli.terrain == "flat"
        and args_cli.height_scan == "mesh"
        and os.path.isfile(f"./envs/{args_cli.custom_env}.usd")
    ):
        height_scanner_env = RayCasterCfg(
            class_type=RayCasterSafeVis,
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=[f"/World/{args_cli.custom_env}_scan"],
        )
    else:
        height_scanner_env = None

    # Mounted only under --lidar_map (see sensors/lidar_elevation_map.py and
    # omniverse_sim.py's --policy_path branch for what reads this). None (disabled)
    # otherwise: a bare list/str/float assigned directly in this class body would be
    # misread by InteractiveScene as a scene entity to instantiate (see
    # _build_mid360_scanner_cfg for why this is a module-level function instead).
    mid360_scanner = _build_mid360_scanner_cfg()

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # ponytail: light the dome with Isaac's bundled studio HDRI (see _STUDIO_HDRI above)
    # so the Go2's shells catch real reflections instead of reading as flat gray.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0), intensity=900.0,
            texture_file=_STUDIO_HDRI or None,
        ),
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (1920, 1080)

    origin_type: Literal["world", "env", "asset_root"] = "world"

    env_index: int = 0

    asset_name: str | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg()
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class UnitreeGo2CustomEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0, 0, 0.4)
        # -90 deg yaw: positive yaw is counter-clockwise from above, so a "turn right" is negative.
        # self.scene.robot.init_state.rot = (math.cos(math.radians(-45)), 0.0, 0.0, math.sin(math.radians(-45)))
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        if self.scene.height_scanner_env is not None:
            self.scene.height_scanner_env.prim_path = "{ENV_REGEX_NS}/Robot/base"
            self.scene.height_scanner_env.update_period = self.decimation * self.sim.dt
            self.observations.policy.height_scan = ObsTerm(
                func=height_scan_merged,
                params={
                    "sensor_cfg": SceneEntityCfg("height_scanner"),
                    "sensor_cfg_2": SceneEntityCfg("height_scanner_env"),
                },
                clip=(-1.0, 1.0),
            )

        if self.scene.mid360_scanner is not None:
            self.scene.mid360_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
            self.scene.mid360_scanner.update_period = self.decimation * self.sim.dt

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class AnagumaCustomEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Tsubame Industries "Anaguma" quadruped (robots/anaguma/) — no walking policy has
    been trained for this robot yet, so actions.joint_pos.scale below is a placeholder:
    update it (and check the observation layout above still matches) once a real policy
    is available, the same way go2_blind_gru_phase4 was wired in via --policy_path.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = ANAGUMA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Anaguma's own root link is named "base_link" (Go2's is "base") — every
        # body_names reference below has to use that name instead.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        if self.scene.height_scanner_env is not None:
            self.scene.height_scanner_env.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
            self.scene.height_scanner_env.update_period = self.decimation * self.sim.dt
            self.observations.policy.height_scan = ObsTerm(
                func=height_scan_merged,
                params={
                    "sensor_cfg": SceneEntityCfg("height_scanner"),
                    "sensor_cfg_2": SceneEntityCfg("height_scanner_env"),
                },
                clip=(-1.0, 1.0),
            )

        # PLACEHOLDER — no trained policy yet; update to match whatever policy is plugged
        # in later (its action scale/offset convention, exactly like unitree_go2's 0.25).
        self.actions.joint_pos.scale = 1.0

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        G1_MINIMAL_CFG = G1_CFG.copy()
        G1_MINIMAL_CFG.spawn.usd_path = "./robots/g1/g1.usd"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.undesired_contacts = None

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link"]
