"""Mid-360 LiDAR sensor, ported from unitree_rl_lab (origin/feat/mid360 branch) for
running its go2_height_map (Go2-Perceptive-Mid360-Phase5-C) policy inside this project.

Upstream is OmniPerception's LidarSensor (https://github.com/aCodeDog/OmniPerception),
carried into unitree_rl_lab and then here. ``LidarSensor`` extends the stock
:class:`isaaclab.sensors.RayCaster`, so nothing needs registering inside isaaclab.
``scan_patterns/mid360.npy`` holds the recorded Mid-360 scan sequence the pattern loader
reads.

``robot_occluder.py`` / ``OccludedRollingLivoxSensor`` were NOT ported: they model the
robot's own body blocking its rays (``MID360_DYNAMIC_MESH``), which upstream itself
defaults off and this project's usage never turns on.
"""

from .lidar_sensor import LidarSensor
from .lidar_sensor_cfg import LidarSensorCfg
from .lidar_sensor_data import LidarSensorData
from .patterns import livox_pattern
from .patterns_cfg import LivoxPatternCfg
from .rolling_livox_sensor import RollingLivoxSensor, RollingLivoxSensorCfg
