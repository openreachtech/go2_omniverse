#!/usr/bin/env python3
"""Open-RMF fleet adapter bridging RMF's traffic negotiation to go2_omniverse's
existing waypoint-follow controller.

Runs under a full system ROS 2 Humble install (where RMF itself lives), separate from
go2_omniverse's own process (Isaac Sim's bundled, Python-only ROS 2 Jazzy rclpy — no
colcon/custom-msg build there). The two talk over plain geometry_msgs/std_msgs/nav_msgs
topics only, since that's all the bundled side can use — see package.xml's description.

Per robot <i>:
  - subscribes to go2_omniverse's existing ``robot<i>/odom`` (nav_msgs/Odometry) for
    position feedback, fed into RMF via RobotUpdateHandle.update_position.
  - publishes ``robot<i>/rmf_target`` (geometry_msgs/PoseStamped, the next point to walk
    to) and ``robot<i>/rmf_hold`` (std_msgs/Bool, true = stand still) — read by
    omniverse_sim.py's --rmf_control branch of update_waypoint_command, which reuses its
    existing proportional heading/speed controller against whatever these say instead of
    a static --waypoints list.

RMF hands each robot a full timed plan (follow_new_path's `waypoints`, each with a
scheduled arrival time) once per dispatched task — already negotiated against every
other robot's reserved itinerary, so a plan that would cross another robot's reserved
slot comes back with a later time on the contested waypoint rather than an explicit
"wait" message. Go2RobotCommand turns that into "stand still" (rmf_hold=True) whenever
it arrives at a plan waypoint before its scheduled time, and only releases the next
target once that time has passed — which is what actually keeps the two robots from
occupying the shared corridor at the same time.
"""

from __future__ import annotations

import datetime
import math
import os
from functools import partial

import json
import uuid

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rmf_task_msgs.msg import ApiRequest, ApiResponse
from std_msgs.msg import Bool

import rmf_adapter as adpt
import rmf_adapter.battery as battery
import rmf_adapter.fleet_update_handle as fleet_update_handle
import rmf_adapter.geometry as geometry
import rmf_adapter.graph as graph
import rmf_adapter.plan as plan
import rmf_adapter.vehicletraits as traits


def _load_config(share_dir: str):
    with open(os.path.join(share_dir, "config", "nav_graph.yaml")) as f:
        nav_cfg = yaml.safe_load(f)
    with open(os.path.join(share_dir, "config", "fleet_config.yaml")) as f:
        fleet_cfg = yaml.safe_load(f)
    return nav_cfg, fleet_cfg


def _build_graph(nav_cfg: dict) -> tuple[graph.Graph, dict[str, int]]:
    g = graph.Graph()
    map_name = nav_cfg["map_name"]
    name_to_index: dict[str, int] = {}
    has_charger = False
    for i, wp in enumerate(nav_cfg["waypoints"]):
        node = g.add_waypoint(map_name, [float(wp["x"]), float(wp["y"])])
        if wp.get("is_charger"):
            node.set_charger(True)
            has_charger = True
        if wp.get("is_holding_point"):
            node.set_holding_point(True)
        name_to_index[wp["name"]] = i
        g.add_key(wp["name"], i)
    assert has_charger, (
        "nav_graph.yaml needs at least one waypoint with is_charger: true — "
        "FleetUpdateHandle::add_robot throws without one, even though nothing here "
        "actually models battery charging."
    )
    for a, b in nav_cfg["lanes"]:
        g.add_bidir_lane(name_to_index[a], name_to_index[b])
    return g, name_to_index


def _yaw_of_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Go2RobotCommand(adpt.RobotCommandHandle):
    """Drives one simulated Go2 through an RMF-assigned, already-negotiated path."""

    def __init__(self, node: Node, robot_name: str, graph_: graph.Graph, start_waypoint: int,
                 waypoint_radius: float):
        adpt.RobotCommandHandle.__init__(self)
        self.node = node
        self.robot_name = robot_name
        self.graph = graph_
        self.waypoint_radius = waypoint_radius
        # Best-known graph waypoint for update_position's topology hint — starts at the
        # robot's configured spawn waypoint, then tracks whichever plan waypoint it's
        # currently heading to or has most recently passed.
        self.nearest_waypoint = start_waypoint

        self.updater = None  # set by add_robot's handle_cb
        self.timer = None
        self.active = False
        self.path = []
        self.path_index = 0
        self.last_xy_yaw = None  # updated by the odom subscription
        self._last_hold_logged = None

        topic_ns = f"robot{robot_name}"
        self.target_pub = node.create_publisher(PoseStamped, f"{topic_ns}/rmf_target", 10)
        self.hold_pub = node.create_publisher(Bool, f"{topic_ns}/rmf_hold", 10)
        node.create_subscription(Odometry, f"{topic_ns}/odom", self._on_odom, 10)
        # Independent of path-following: RMF needs to know where idle robots are too.
        node.create_timer(0.5, self._report_position)

    def _report_position(self):
        # test_loop.py's reference RobotCommandHandle never calls update_position at
        # all — only update_current_waypoint, once per graph node, while actively
        # following a path (see _tick below). update_position's sibling methods
        # (update_lost_position, update_off_grid_position) suggest it's meant for "where
        # is this robot, I'm not sure" localization, not continuous telemetry: calling it
        # every 0.5s with our controller's naturally fluctuating live orientation looked
        # exactly like a robot continuously going off-plan, triggering constant
        # replanning. Only report it while idle, where update_current_waypoint never
        # fires on its own.
        if self.active:
            return
        if self.updater is None or self.last_xy_yaw is None:
            return
        x, y, yaw = self.last_xy_yaw
        now = datetime.datetime.fromtimestamp(self.node.get_clock().now().nanoseconds / 1e9)
        self.updater.update_position(
            [plan.Start(now, self.nearest_waypoint, yaw, location=np.array([[x], [y]]))]
        )

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = _yaw_of_quat(msg.pose.pose.orientation)
        self.last_xy_yaw = (p.x, p.y, yaw)

    def _publish_target(self, x: float, y: float, yaw: float, hold: bool):
        pose = PoseStamped()
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.target_pub.publish(pose)
        self.hold_pub.publish(Bool(data=hold))

    # -- adpt.RobotCommandHandle overrides -----------------------------------------
    def follow_new_path(self, waypoints, next_arrival_estimator, path_finished_callback):
        # RMF re-issues follow_new_path roughly every ~10s even for a robot that's
        # mid-leg and in no conflict (observed directly: it happens on the same cadence
        # whether the robot is idle, holding, or walking, and set_infinite_delay — which
        # rules out schedule-drift replanning — didn't change that). Resetting
        # path_index/the timer on every one of these meant a robot going for its final,
        # longest leg (several seconds) got restarted from scratch before ever arriving.
        # If the new path's destination hasn't changed, assume this is one of those
        # routine re-issues and keep the current leg running uninterrupted instead of
        # restarting it — only a genuinely new destination (a new task) gets a full
        # reset.
        # NOT conditioned on self.active: RMF's own RobotCommandHandle machinery calls
        # our stop() (which sets active=False) immediately before *every*
        # follow_new_path, including routine same-destination re-issues — confirmed by
        # logging follow_new_path's actual (active, old_goal, new_goal) tuples live:
        # active was False every single time, even mid-leg, so an active-gated check
        # never once matched and path_index reset to 0 on every ~8-10s re-issue. self.path
        # itself isn't touched by stop(), so comparing goals through it still works.
        new_goal = tuple(round(float(v), 3) for v in waypoints[-1].position[:2])
        old_goal = tuple(round(float(v), 3) for v in self.path[-1].position[:2]) if self.path else None
        same_goal = bool(self.path) and old_goal == new_goal
        # Cancel the old timer either way — it closes over the previous call's
        # next_arrival_estimator/path_finished_callback, which may no longer be valid
        # once RMF has issued a new pair, so we always rebind to the fresh ones below.
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if not same_goal:
            print(f"[go2_rmf_adapter] robot {self.robot_name}: new path, "
                  f"{len(waypoints)} waypoint(s)", flush=True)
            self.path_index = 0
        else:
            self.path_index = min(self.path_index, len(waypoints) - 1)
        self.path = waypoints
        self.active = True
        self.timer = self.node.create_timer(
            0.2, partial(self._tick, next_arrival_estimator=next_arrival_estimator,
                         path_finished_callback=path_finished_callback),
        )

    def stop(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.active = False
        if self.last_xy_yaw is not None:
            x, y, yaw = self.last_xy_yaw
            self._publish_target(x, y, yaw, hold=True)

    def dock(self, dock_name, docking_finished_callback):
        # No docking lanes in this graph.
        docking_finished_callback()

    def _tick(self, next_arrival_estimator, path_finished_callback):
        if not self.active or self.last_xy_yaw is None:
            return
        if self.path_index >= len(self.path):
            self.active = False
            if self.timer is not None:
                self.timer.cancel()
            path_finished_callback()
            return

        wp = self.path[self.path_index]
        tx, ty = float(wp.position[0]), float(wp.position[1])
        tyaw = float(wp.position[2]) if len(wp.position) > 2 else 0.0
        now = datetime.datetime.fromtimestamp(self.node.get_clock().now().nanoseconds / 1e9)
        scheduled = wp.time
        early = scheduled > now

        x, y, _ = self.last_xy_yaw
        arrived = math.hypot(tx - x, ty - y) < self.waypoint_radius

        # Stand still either because we got here before our negotiated slot, or because
        # we haven't arrived yet (the sim-side controller walks toward rmf_target on its
        # own — hold=False there just means "go", not "teleport").
        if early != self._last_hold_logged:
            print(f"[go2_rmf_adapter] robot {self.robot_name}: "
                  f"{'HOLD (waiting for negotiated slot)' if early else 'proceed'} "
                  f"-> waypoint {self.path_index} ({tx:.2f}, {ty:.2f})", flush=True)
            self._last_hold_logged = early
        self._publish_target(tx, ty, tyaw, hold=early)

        if wp.graph_index is not None:
            self.nearest_waypoint = wp.graph_index
            if self.updater is not None:
                self.updater.update_current_waypoint(wp.graph_index, tyaw)

        if arrived and not early:
            print(f"[go2_rmf_adapter] robot {self.robot_name}: reached waypoint "
                  f"{self.path_index} ({tx:.2f}, {ty:.2f})", flush=True)
            self.path_index += 1
            self._last_hold_logged = None
            if self.path_index < len(self.path):
                next_wp = self.path[self.path_index]
                next_arrival_estimator(self.path_index, next_wp.time - now)

    def set_updater(self, updater):
        self.updater = updater
        # Our _tick's hold/proceed logic already paces each robot against its
        # negotiated waypoint times by itself (see the `early` check above). RMF's own
        # separate schedule-drift replanning (triggered whenever actual progress lags
        # the plan's precise timing model by more than FleetUpdateHandle's maximum
        # delay) was firing every ~10s regardless of real progress — this simple
        # 0.2s-granularity controller never tracks the plan's timing tightly enough to
        # satisfy it, so robots got reset back to square one before ever finishing a
        # leg. Disabling it here (infinite delay) leaves negotiation/holding intact;
        # only the *redundant* drift-triggered replanning goes away.
        updater.set_infinite_delay()


def main():
    rclpy.init()
    try:
        adpt.init_rclcpp()
    except RuntimeError:
        pass

    share_dir = get_package_share_directory("go2_rmf_adapter")
    nav_cfg, fleet_cfg = _load_config(share_dir)
    map_name = nav_cfg["map_name"]

    g, name_to_index = _build_graph(nav_cfg)

    profile = traits.Profile(geometry.make_final_convex_circle(fleet_cfg["footprint_radius"]),
                              geometry.make_final_convex_circle(fleet_cfg["vicinity_radius"]))
    vehicle_traits = traits.VehicleTraits(
        linear=traits.Limits(fleet_cfg["linear_velocity"], fleet_cfg["linear_acceleration"]),
        angular=traits.Limits(fleet_cfg["angular_velocity"], fleet_cfg["angular_acceleration"]),
        profile=profile,
    )

    adapter = adpt.Adapter.make("go2_fleet_adapter")
    # add_fleet's 4th param is an optional rmf-web API server URI, not the map name —
    # map_name only matters per-waypoint (already baked into _build_graph's
    # add_waypoint calls) and for dispatch_task's place-name lookups.
    fleet = adapter.add_fleet(fleet_cfg["fleet_name"], vehicle_traits, g)

    def patrol_req_cb(json_desc):
        confirmation = fleet_update_handle.Confirmation()
        confirmation.accept()
        return confirmation

    fleet.consider_patrol_requests(patrol_req_cb)

    battery_sys = battery.BatterySystem.make(24.0, 40.0, 8.8)
    mech_sys = battery.MechanicalSystem.make(70.0, 40.0, 0.22)
    motion_sink = battery.SimpleMotionPowerSink(battery_sys, mech_sys)
    ambient_sink = battery.SimpleDevicePowerSink(battery_sys, battery.PowerSystem.make(20.0))
    tool_sink = battery.SimpleDevicePowerSink(battery_sys, battery.PowerSystem.make(10.0))
    ok = fleet.set_task_planner_params(
        battery_sys, motion_sink, ambient_sink, tool_sink, 0.2, 1.0, False
    )
    assert ok, "set_task_planner_params failed"

    cmd_node = Node("go2_rmf_robot_commands")
    # rmf_task_dispatcher's task_api_requests/responses topics use RELIABLE +
    # TRANSIENT_LOCAL (so a late-joining dispatcher still sees the last request/response)
    # — a default VOLATILE publisher is QoS-incompatible and silently never delivers.
    api_qos = QoSProfile(
        depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    api_request_pub = cmd_node.create_publisher(ApiRequest, "/task_api_requests", api_qos)
    cmd_node.create_subscription(
        ApiResponse, "/task_api_responses",
        lambda msg: print(f"[go2_rmf_adapter] task_api_response {msg.request_id}: {msg.json_msg}", flush=True),
        api_qos,
    )
    robot_cmds = []

    for robot_cfg in nav_cfg["robots"]:
        name = robot_cfg["name"]
        start_idx = name_to_index[robot_cfg["start_waypoint"]]
        cmd = Go2RobotCommand(cmd_node, name, g, start_idx, fleet_cfg["waypoint_radius"])
        robot_cmds.append((name, cmd, robot_cfg["goal_waypoint"]))

        starts = [plan.Start(adapter.now(), start_idx, 0.0)]
        fleet.add_robot(cmd, name, profile, starts, cmd.set_updater)

    adapter.start()
    print(f"[go2_rmf_adapter] fleet '{fleet_cfg['fleet_name']}' started with "
          f"{len(robot_cmds)} robot(s)", flush=True)

    def dispatch_one(name, goal_name):
        request_id = f"go2_rmf.{name}.{goal_name}.{uuid.uuid4().hex[:8]}"
        # robot_task_request (rmf_api_msgs/schemas/robot_task_request.json), not
        # dispatch_task_request: the latter auctions the task to whichever robot in
        # the fleet bids cheapest, which for this symmetric graph swapped robot 0 and
        # 1's destinations (each got sent to its own *nearer* end instead of crossing
        # the shared corridor) — defeating the point of this demo. robot_task_request
        # pins the task to the robot we actually intend, by name.
        payload = {
            "type": "robot_task_request",
            "robot": name,
            "fleet": fleet_cfg["fleet_name"],
            "request": {
                "category": "patrol",
                "unix_millis_earliest_start_time": 0,
                "description": {"places": [goal_name], "rounds": 1},
            },
        }
        api_request_pub.publish(ApiRequest(json_msg=json.dumps(payload), request_id=request_id))
        print(f"[go2_rmf_adapter] dispatch request {request_id}: robot {name} -> {goal_name}",
              flush=True)

    # Staggered, not simultaneous: dispatching every robot in the same instant gave two
    # robots the same distance from wp_center, at the same speed, exactly tied arrival
    # times — too close a tie for the negotiator to treat as a real conflict (both went
    # straight through; see fleet_config.yaml's comment). Giving the first robot a clear
    # few-second head start removes the tie, so the second robot's plan has an
    # unambiguous "someone is already there" to hold for.
    dispatch_stagger_s = 4.0

    _stagger_timers = {}

    def _dispatch_once_and_cancel(name, goal_name):
        _stagger_timers.pop(name).cancel()
        dispatch_one(name, goal_name)

    def dispatch_patrol_tasks():
        # A real rmf_task_dispatcher (unlike MockAdapter's in-process dispatch_task
        # shortcut) only has something to dispatch to once it's discovered this fleet
        # over ROS 2 pub/sub — this timer is a one-shot delay for that discovery rather
        # than a recurring poll (cancelled on its first call).
        dispatch_timer.cancel()
        for i, (name, cmd, goal_name) in enumerate(robot_cmds):
            if i == 0:
                dispatch_one(name, goal_name)
                continue
            _stagger_timers[name] = cmd_node.create_timer(
                i * dispatch_stagger_s, partial(_dispatch_once_and_cancel, name, goal_name)
            )

    dispatch_timer = cmd_node.create_timer(3.0, dispatch_patrol_tasks)

    executor = MultiThreadedExecutor()
    executor.add_node(cmd_node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cmd_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
