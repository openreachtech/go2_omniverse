"""Script to play a checkpoint if an RL agent from RSL-RL."""
from __future__ import annotations


"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher


import cli_args  
import time
import os
import threading


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--custom_env", type=str, default="office", help="Setup the environment")
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument("--terrain", type=str, default="rough", help="Setup the robot")
parser.add_argument(
    "--height_scan", type=str, default="mesh", choices=["mesh", "ground"],
    help="height_scan source when a custom env is loaded: 'mesh' merges in the custom "
         "env's geometry (see height_scan_merged in custom_rl_env.py), 'ground' always "
         "reads /World/ground only, ignoring the custom env's mesh.",
)
parser.add_argument(
    "--height_scan_vis", type=str, default="on", choices=["on", "off"],
    help="Show the height_scan marker grid (the merged/ground result — see height_scan "
         "above) as red spheres in the viewport. 'off' skips drawing them entirely.",
)
parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")
parser.add_argument("--twinbot", action="store_true", default=False,
                    help="Digital-twin mode: drive sim joints from real Go2 via /real_dog/joint_states "
                         "(requires twinbot_bridge.py running on the Jetson)")
parser.add_argument("--capture", type=int, default=0,
                    help="Capture N settle frames then write hero PNGs from several angles and exit "
                         "(headless-safe; uses an isaaclab Camera render product, not a window grab).")
parser.add_argument("--capture_dir", type=str, default="/tmp/twin_hero",
                    help="Directory to write hero PNGs into when --capture > 0.")
parser.add_argument(
    "--policy_path", type=str, default=None,
    help="Path to an exported TorchScript policy (rsl_rl's <run>/exported/policy.pt), "
         "e.g. from a blind/recurrent policy trained outside this repo. When set, this "
         "replaces the usual checkpoint-search + MLP-reconstruction path (agent_cfg.py's "
         "experiment_name/load_run/load_checkpoint) entirely, and observations are built "
         "directly to match that export instead of this repo's own ObservationsCfg. "
         "Only tested with --robot_amount 1 (the exported module's GRU hidden state is "
         "sized for a single environment).",
)
parser.add_argument(
    "--waypoints", type=str, default=None,
    help="Semicolon-separated x,y pairs in world coordinates to walk through in order, "
         "e.g. '1,0;2,2;0,3'. Overrides keyboard control every step: base_command is set "
         "each frame by steering toward the current waypoint (see _update_waypoint_command "
         "in run_sim()) instead of waiting for WASD input. Robot must be at env_cfg.scene."
         "robot's ENV_REGEX_NS origin's coordinate frame (env 0's world origin) — with a "
         "single env this is just plain world x,y.",
)
parser.add_argument(
    "--waypoint_radius", type=float, default=0.3,
    help="Distance (m) to a waypoint at which it's considered reached and the robot "
         "advances to the next one.",
)
parser.add_argument(
    "--waypoint_speed", type=float, default=0.8,
    help="Max forward speed (m/s) commanded while walking toward a waypoint.",
)
parser.add_argument(
    "--waypoint_loop", action="store_true", default=False,
    help="Cycle back to the first waypoint after reaching the last one, instead of "
         "stopping there.",
)
parser.add_argument(
    "--waypoint_delay", type=float, default=10.0,
    help="Seconds to stand still (waypoint markers still draw, command stays zero) after "
         "entering the main loop before walking toward the first waypoint — gives the "
         "livestream viewport time to finish loading before the robot starts moving.",
)


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _ckpt(msg: str):
    print(f"[go2_omniverse] {time.strftime('%H:%M:%S')} {msg}", flush=True)


_ckpt("AppLauncher: constructing...")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_ckpt("AppLauncher: ready")


import omni
_ckpt("import omni: ok")


ext_manager = omni.kit.app.get_app().get_extension_manager()
# Required OmniGraph + ROS 2 bridge extensions. We use the non-immediate
# set_extension_enabled + simulation_app.update() pump: set_extension_enabled_immediate
# was observed to deadlock on isaacsim.core.nodes on Isaac Sim 5.0 / Ubuntu 24.04.
# isaacsim.ros2.bridge depends on isaacsim.core.nodes so enabling the bridge
# transitively enables the nodes extension.
# NOTE: we deliberately do NOT enable `isaacsim.ros2.bridge`. That extension
# loads its own in-process rcl_interfaces typesupport which conflicts with the
# rclpy we use from Python and triggers a ParameterEvent assertion. Since this
# project publishes ROS 2 data through rclpy from omniverse_sim.py directly
# (not via OmniGraph ROS2Helper nodes), we only need the OmniGraph core
# extensions for the action-graph based camera stream.
_required_exts = (
    "omni.graph.core",
    "omni.graph.action",
    "omni.graph.nodes",
    # Needed because ros2.py imports isaacsim.sensors.rtx.LidarRtx. This is
    # transitively enabled by isaacsim.ros2.bridge, but we don't enable the
    # bridge (see note above).
    "isaacsim.sensors.rtx",
)
for _ext in _required_exts:
    _ckpt(f"requesting extension: {_ext}")
    ext_manager.set_extension_enabled(_ext, True)

# Pump frames until all requested extensions are actually enabled (or give up).
_t0 = time.time()
while time.time() - _t0 < 120:
    simulation_app.update()
    pending = [e for e in _required_exts if not ext_manager.is_extension_enabled(e)]
    if not pending:
        break
    time.sleep(0.05)
still_pending = [e for e in _required_exts if not ext_manager.is_extension_enabled(e)]
_ckpt(f"extensions enabled in {time.time()-_t0:.2f}s (still_pending={still_pending})")

# FOR VR SUPPORT
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.core", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.steamvr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.simulatedxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.openxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.telemetry", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.profile.vr", True)


"""Rest everything follows."""
_ckpt("import gymnasium")
import gymnasium as gym
_ckpt("import torch")
import torch
_ckpt("import carb")
import carb


_ckpt("import isaaclab_tasks.utils.parse_cfg")
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path
_ckpt("import isaaclab_rl.rsl_rl")
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
_ckpt("import isaaclab.sim")
import isaaclab.sim as sim_utils
_ckpt("import omni.appwindow")
import omni.appwindow


_ckpt("import rclpy")
import rclpy
_ckpt("import ros2 module")
from ros2 import RobotBaseNode, add_camera, add_rtx_lidar, pub_robo_data_ros2
from geometry_msgs.msg import Twist


_ckpt("import agent_cfg")
from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
_ckpt("import custom_rl_env (this pulls isaaclab_assets Unitree USD cfg)")
from custom_rl_env import UnitreeGo2CustomEnvCfg, G1RoughEnvCfg
import custom_rl_env
_ckpt("import omnigraph")
from omnigraph import create_front_cam_omnigraph
_ckpt("all imports complete")

# twinbot import is deferred until after rclpy.init() in run_sim()


def _load_mlp_policy(ckpt_path: str, hidden_dims, activation_name: str, device: str):
    """Load an MLP actor from a legacy rsl_rl ActorCritic checkpoint.

    The installed rsl_rl-lib (5.x) has a different config/load API than the one
    used to train the shipped checkpoints (pre-2025). The checkpoint only
    contains MLP weights for actor / critic plus a learned std, so we rebuild a
    matching nn.Sequential for inference and skip the runner entirely.
    """
    import torch.nn as nn

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model_state_dict"]

    # Derive input dim from first layer
    actor_in = sd["actor.0.weight"].shape[1]
    actor_out = sd["actor.6.weight"].shape[0]

    activation = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation_name.lower()]

    layers = []
    dims = [actor_in, *hidden_dims]
    for i in range(len(hidden_dims)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dims[-1], actor_out))
    actor = nn.Sequential(*layers)

    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.to(device).eval()
    return actor, actor_in, actor_out


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env.base_command["0"] = [0, 0, -1]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == 'I':
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_rl_env.base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True

import os

def setup_custom_env():
    if args_cli.terrain != 'flat':
        return
    usd_path = f"./envs/{args_cli.custom_env}.usd"
    if not os.path.isfile(usd_path):
        print(f"[go2_omniverse] custom env usd not found: {usd_path}")
        return
    try:
        cfg_scene = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg_scene.func(f"/World/{args_cli.custom_env}", cfg_scene, translation=(0.0, 0.0, 0.0))

        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd
        import numpy as np

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(f"/World/{args_cli.custom_env}")
        mesh_prims = []
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                mesh_prims.append(prim)

        # RayCaster (isaaclab/sensors/ray_caster/ray_caster.py) only reads the FIRST Mesh
        # prim it finds under a given path — it does not merge a subtree with several
        # separate Mesh prims (common in these envs: floor, ramp steps, walls as distinct
        # meshes). So height_scanner_env has nothing to point at that reliably covers the
        # whole scene; build one combined Mesh prim in world space for it to use instead.
        # Skipped under --height_scan ground: height_scanner_env doesn't exist then (see
        # custom_rl_env.py), so this mesh would just be dead weight.
        all_points = []
        all_tris = []
        offset = 0
        for prim in (mesh_prims if args_cli.height_scan == "mesh" else []):
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get())
            if pts.size == 0:
                continue
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            transform = np.array(omni.usd.get_world_transform_matrix(prim)).T
            pts_world = np.matmul(pts, transform[:3, :3].T) + transform[:3, 3]

            # Fan-triangulate every face. convert_to_warp_mesh() (isaaclab/utils/warp/ops.py,
            # used by RayCaster to build its collision mesh) requires pure triangle indices,
            # but USD box/cube primitives are commonly authored as quads (4 verts/face) —
            # feeding those straight through silently reinterprets every 3 raw index values
            # as one triangle, scrambling faces into garbage geometry with the right bbox
            # but wrong internal shape (this is what caused the ~5-10cm phantom bumps).
            idx = 0
            for c in counts:
                c = int(c)
                face = face_indices[idx: idx + c]
                for i in range(1, c - 1):
                    all_tris.append((face[0] + offset, face[i] + offset, face[i + 1] + offset))
                idx += c

            all_points.append(pts_world)
            offset += len(pts)

        if all_points:
            scan_mesh_path = f"/World/{args_cli.custom_env}_scan"
            scan_mesh = UsdGeom.Mesh.Define(stage, scan_mesh_path)
            merged_tris = np.array(all_tris, dtype=np.int64)
            scan_mesh.CreatePointsAttr(np.concatenate(all_points).tolist())
            scan_mesh.CreateFaceVertexCountsAttr([3] * len(merged_tris))
            scan_mesh.CreateFaceVertexIndicesAttr(merged_tris.flatten().tolist())
            UsdGeom.Imageable(scan_mesh.GetPrim()).MakeInvisible()
    except Exception as e:
        print(f"[go2_omniverse] Error loading custom environment '{args_cli.custom_env}': {e}")


def capture_hero_shots(env, policy, obs, device, n_settle, out_dir):
    """Write hero PNGs of the Go2 from several angles via an isaaclab Camera render
    product. Headless-safe: it reads the RGB tensor directly, so it does NOT depend on
    an on-screen Kit window (this Isaac build renders offscreen). The policy keeps the
    robot balancing while we shoot."""
    import os
    import numpy as np
    from PIL import Image
    from isaaclab.sensors import Camera, CameraCfg

    os.makedirs(out_dir, exist_ok=True)

    # Drop the robot into a REAL environment (NVIDIA's stock warehouse) streamed from the
    # Isaac asset server, instead of the empty HDRI void. This is the single biggest
    # realism lever: real geometry, real bounce light, real reflections around the dog.
    try:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path() or \
            "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
        wh = f"{root}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        sim_utils.UsdFileCfg(usd_path=wh).func("/World/hero_env", sim_utils.UsdFileCfg(usd_path=wh))
        _ckpt(f"loaded hero environment: {wh}")
        # The warehouse brings its own ceiling lighting; dim our flat fill dome + sun so the
        # scene reads naturally (path the studio HDRI down, not off, for soft reflections).
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for path, val in (("/World/skyLight", 80.0), ("/World/light", 250.0)):
            p = stage.GetPrimAtPath(path)
            if p and p.IsValid():
                a = p.GetAttribute("inputs:intensity")
                if a:
                    a.Set(val)
        in_env = True
    except Exception as e:
        _ckpt(f"hero environment load failed ({type(e).__name__}: {e}) — shooting on HDRI void")
        in_env = False
    # World-anchored hero cam: 1080p, 35mm (less wide/distorted than the 24mm FPV cam).
    cam = Camera(CameraCfg(
        prim_path="/World/hero_cam",
        height=1080, width=1920, update_period=0.0, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e6)),
    ))

    dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0))

    def step():
        nonlocal obs
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))

    # The Camera was created AFTER sim play, so its PHYSICS_READY init callback never
    # fired. Step once so the render product exists, then force-initialize it.
    step()
    if not cam.is_initialized:
        cam._initialize_impl()
        cam._is_initialized = True

    # Kill the RL velocity-command debug markers (the floating blue/green arrows) so
    # they don't photobomb the hero shots.
    try:
        env.unwrapped.command_manager.set_debug_vis(False)
    except Exception as e:
        _ckpt(f"could not disable command debug_vis ({type(e).__name__}: {e})")

    # Let the policy stand the robot up and settle.
    for _ in range(max(n_settle, 1)):
        step()
    cam.update(dt)

    # robot base position so shots frame wherever it ended up
    base = _to_numpy_safe(env.unwrapped.scene["robot"].data.root_state_w)[0, :3]
    bx, by, bz = float(base[0]), float(base[1]), float(base[2])
    # aim at the robot's body centre (base sits ~0.4 m up; legs reach the floor)
    tgt = [bx, by, bz - 0.08]
    shots = {
        "hero_front34": [bx + 2.0, by + 1.6, bz + 0.45],   # 3/4 front
        "hero_side":    [bx + 0.10, by + 2.4, bz + 0.10],  # side profile
        "hero_low":     [bx + 1.7, by + 1.0, bz - 0.18],   # low dramatic
    }
    eyes = torch.tensor([shots[k] for k in shots], dtype=torch.float32, device=device)
    targets = torch.tensor([tgt for _ in shots], dtype=torch.float32, device=device)

    saved = []
    for i, name in enumerate(shots):
        cam.set_world_poses_from_view(eyes[i:i + 1], targets[i:i + 1])
        # Re-render a few frames so RT2 accumulation/exposure settles for this view.
        for _ in range(24):
            step()
            cam.update(dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(rgb[..., :3]).save(path)
        saved.append(path)
        _ckpt(f"captured {path}")
    return saved


def _to_numpy_safe(arr):
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            return arr.detach().cpu().numpy()
    import numpy as np
    return np.asarray(arr)


def cmd_vel_cb(msg, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_rl_env.base_command[str(num_robot)] = [x, y, z]



def add_cmd_sub(num_envs):
    node_test = rclpy.create_node('position_velocity_publisher')
    for i in range(num_envs):
        node_test.create_subscription(Twist, f'robot{i}/cmd_vel', lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10)
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()



def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]
def run_sim():
    
    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Play with RSL-RL agent."""
    # parse configuration
    
    env_cfg = UnitreeGo2CustomEnvCfg()

    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()

    # add N robots to env
    env_cfg.scene.num_envs = args_cli.robot_amount

    specify_cmd_for_robots(env_cfg.scene.num_envs)

    agent_cfg = unitree_go2_agent_cfg

    if args_cli.robot == "g1":
        agent_cfg = unitree_g1_agent_cfg

    # Must run before gym.make(): height_scanner_env (initialized during env creation)
    # resolves its mesh_prim_paths immediately and errors out if the custom env's prim
    # doesn't exist in the stage yet.
    setup_custom_env()

    # create isaac environment
    _ckpt(f"gym.make task={args_cli.task} num_envs={env_cfg.scene.num_envs}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    _ckpt("gym.make: done")
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    _ckpt("RslRlVecEnvWrapper: wrapped")
    device = str(env.unwrapped.device)

    if args_cli.policy_path:
        # Exported TorchScript policy (e.g. a blind/recurrent run from a different
        # training repo) — its own observation layout replaces this repo's ObservationsCfg
        # entirely, so it's built by hand below instead of reading env.get_observations().
        _ckpt(f"loading exported policy: {args_cli.policy_path}")
        policy_module = torch.jit.load(args_cli.policy_path, map_location=device)
        policy_module.eval()
        _ckpt("exported policy loaded")

        robot = env.unwrapped.scene["robot"]
        num_envs = env_cfg.scene.num_envs
        if num_envs != 1:
            print(
                f"[go2_omniverse] WARNING: --policy_path's exported GRU hidden_state is "
                f"sized for 1 env, got --robot_amount {num_envs}. Only the first env's "
                f"actions will be sensible."
            )
        num_actions = robot.data.default_joint_pos.shape[1]
        last_action_buf = torch.zeros(num_envs, num_actions, device=device)

        def policy(obs_dict):
            # Order/scale matches this export's deploy.yaml exactly: base_ang_vel (x0.2),
            # projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel (x0.05),
            # last_action. No height_scan — this policy is blind by design.
            cmd = torch.tensor(
                [custom_rl_env.base_command[str(i)] for i in range(num_envs)],
                dtype=torch.float32, device=device,
            )
            obs = torch.cat(
                [
                    robot.data.root_ang_vel_b * 0.2,
                    robot.data.projected_gravity_b,
                    cmd,
                    robot.data.joint_pos - robot.data.default_joint_pos,
                    (robot.data.joint_vel - robot.data.default_joint_vel) * 0.05,
                    last_action_buf,
                ],
                dim=-1,
            ).clamp(-100.0, 100.0)
            action = policy_module(obs)
            last_action_buf.copy_(action)
            return action
    else:
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        # Legacy checkpoint — build a matching MLP for inference (see helper above).
        actor, _, _ = _load_mlp_policy(
            resume_path,
            hidden_dims=agent_cfg["policy"]["actor_hidden_dims"],
            activation_name=agent_cfg["policy"]["activation"],
            device=device,
        )

        def policy(obs_dict):
            obs_tensor = obs_dict["policy"] if hasattr(obs_dict, "__getitem__") and "policy" in obs_dict else obs_dict
            return actor(obs_tensor)

    # reset environment
    _ckpt("env.get_observations()")
    obs = env.get_observations()
    _ckpt("env.get_observations: done")

    # initialize ROS2 node
    _ckpt("rclpy.init + RobotBaseNode")
    rclpy.init()
    base_node = RobotBaseNode(env_cfg.scene.num_envs)
    add_cmd_sub(env_cfg.scene.num_envs)
    _ckpt("ROS2 publishers up")

    twin = None
    if args_cli.twinbot:
        from twinbot import TwinbotSubscriber
        twin = TwinbotSubscriber(env)
        _ckpt("TwinbotSubscriber ready — waiting for /real_dog/joint_states")

    # Lidar disabled pending Unitree_L1.json update for Isaac Sim 5.0 schema.
    annotator_lst = []
    try:
        add_camera(env_cfg.scene.num_envs, args_cli.robot)
        _ckpt("camera added")
    except Exception as e:
        _ckpt(f"add_camera skipped ({type(e).__name__}: {e}) — isaaclab.sensors.Camera API changed in 0.54.x")

    # ROS 2 camera OmniGraph stream requires isaacsim.ros2.bridge, which we
    # deliberately do not enable (see extension-enable note above). Skip.
    _ckpt("camera omnigraph skipped (bridge extension disabled for rclpy compat)")
    _ckpt("entering main loop")

    # Both height_scanner and height_scanner_env have debug_vis off (see custom_rl_env.py) —
    # draw the merged result ourselves instead of two independently-drawn, overlapping grids.
    # --height_scan_vis off skips this whole block, so no markers ever get created/drawn.
    height_vis = None
    height_scanner_sensor = env.unwrapped.scene.sensors.get("height_scanner")
    height_scanner_env_sensor = env.unwrapped.scene.sensors.get("height_scanner_env")
    if args_cli.height_scan_vis == "on" and height_scanner_sensor is not None:
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

        height_vis = VisualizationMarkers(RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/HeightScanMerged"))

    def _draw_merged_height_scan():
        hits = height_scanner_sensor.data.ray_hits_w
        if height_scanner_env_sensor is not None:
            hits_env = height_scanner_env_sensor.data.ray_hits_w
            # higher surface wins: larger world z = closer to a downward-looking sensor.
            use_env = (~torch.isinf(hits_env[..., 2])) & (
                torch.isinf(hits[..., 2]) | (hits_env[..., 2] > hits[..., 2])
            )
            hits = torch.where(use_env.unsqueeze(-1), hits_env, hits)
        points = hits.reshape(-1, 3)
        points = points[~torch.any(torch.isinf(points), dim=1)]
        if points.numel() > 0:
            height_vis.visualize(points)

    # --waypoints overrides keyboard control every step: base_command is recomputed each
    # frame from the robot's actual world pose (simple proportional heading/speed
    # controller — not a path planner, just enough to walk through a fixed list of
    # points instead of driving WASD by hand).
    update_waypoint_command = None
    if args_cli.waypoints:
        waypoints = [tuple(map(float, pair.split(","))) for pair in args_cli.waypoints.split(";")]
        print(f"[go2_omniverse] waypoints: {waypoints}")
        waypoint_robot = env.unwrapped.scene["robot"]
        num_wp_envs = env_cfg.scene.num_envs
        waypoints_t = torch.tensor(waypoints, dtype=torch.float32, device=device)
        waypoint_idx = torch.zeros(num_wp_envs, dtype=torch.long, device=device)
        last_wp = len(waypoints) - 1

        # Static markers at each waypoint (z fixed just above ground level — good enough
        # for the flat custom envs this is used with): grey = already passed, green =
        # current target, blue = still ahead. Re-drawn every step so the colors track
        # waypoint_idx as env 0 (the only one considered here) advances.
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        waypoint_vis = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Waypoints",
                markers={
                    "pending": sim_utils.SphereCfg(
                        radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0))
                    ),
                    "current": sim_utils.SphereCfg(
                        radius=0.12, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1))
                    ),
                    "done": sim_utils.SphereCfg(
                        radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
                    ),
                },
            )
        )
        waypoint_translations = torch.cat(
            [waypoints_t, torch.full((len(waypoints), 1), 0.05, device=device)], dim=-1
        )

        def draw_waypoints():
            idx = int(waypoint_idx[0].item())
            marker_indices = [2 if i < idx else 1 if i == idx else 0 for i in range(len(waypoints))]
            waypoint_vis.visualize(waypoint_translations, marker_indices=torch.tensor(marker_indices, device=device))

        def update_waypoint_command():
            draw_waypoints()
            if time.time() - start_time < args_cli.waypoint_delay:
                for i in range(num_wp_envs):
                    custom_rl_env.base_command[str(i)] = [0.0, 0.0, 0.0]
                return
            pos = waypoint_robot.data.root_pos_w[:, :2]
            heading = waypoint_robot.data.heading_w
            target = waypoints_t[waypoint_idx]
            delta = target - pos
            dist = torch.linalg.norm(delta, dim=-1)
            desired_heading = torch.atan2(delta[:, 1], delta[:, 0])
            heading_error = torch.atan2(
                torch.sin(desired_heading - heading), torch.cos(desired_heading - heading)
            )

            arrived = dist < args_cli.waypoint_radius
            finished = arrived & (waypoint_idx == last_wp) if not args_cli.waypoint_loop else torch.zeros_like(arrived)
            if args_cli.waypoint_loop:
                waypoint_idx[arrived] = (waypoint_idx[arrived] + 1) % len(waypoints)
            else:
                waypoint_idx[arrived] = torch.clamp(waypoint_idx[arrived] + 1, max=last_wp)

            # slow down inside the arrival radius instead of overshooting at full speed
            speed_cap = torch.clamp(
                dist / max(args_cli.waypoint_radius, 1e-3) * args_cli.waypoint_speed,
                max=args_cli.waypoint_speed,
            )
            lin_vel_x = speed_cap * torch.clamp(torch.cos(heading_error), min=0.0)
            ang_vel_z = torch.clamp(heading_error * 2.0, -1.2, 1.2)
            lin_vel_x = torch.where(finished, torch.zeros_like(lin_vel_x), lin_vel_x)
            ang_vel_z = torch.where(finished, torch.zeros_like(ang_vel_z), ang_vel_z)

            for i in range(num_wp_envs):
                custom_rl_env.base_command[str(i)] = [lin_vel_x[i].item(), 0.0, ang_vel_z[i].item()]

    if args_cli.capture > 0:
        capture_hero_shots(env, policy, obs, device, args_cli.capture, args_cli.capture_dir)
        env.close()
        simulation_app.close()
        return

    start_time = time.time()
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            if update_waypoint_command is not None:
                update_waypoint_command()
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if height_vis is not None:
                _draw_merged_height_scan()
            if twin is not None:
                # Overwrite physics-stepped state with the real dog's state.
                # Kinematic playback — bypasses PD/gravity for an exact mirror.
                twin.apply(device)
            pub_robo_data_ros2(args_cli.robot, env_cfg.scene.num_envs, base_node, env, annotator_lst, start_time)
    env.close()
