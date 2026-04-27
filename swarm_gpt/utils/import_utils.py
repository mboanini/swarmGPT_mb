"""Import utility package to deal with any crazyswarm2 import issues."""

import logging
import sys

from swarm_gpt.utils.utils import get_ros_package_path

logger = logging.getLogger(__name__)


try:
    import crazyflie_py as pycrazyswarm  # noqa: F401
except ImportError:
    path = get_ros_package_path("crazyswarm2", heuristic_search=True)
    crazyflie_py_path = path / "scripts"
    if str(crazyflie_py_path) not in sys.path:
        sys.path.insert(0, str(crazyflie_py_path))

    import crazyflie_py as pycrazyswarm  # noqa: F401

try:
    from crazyflie_interfaces.msg import Position  # noqa: F401
except ImportError:
    # Mock the import of crazyflie_interfaces in case we are only running in sim, i.e. without ROS 2
    logger.warning("crazyflie_interfaces not installed. Mocking import with simple namespace object.")
    Position = None
