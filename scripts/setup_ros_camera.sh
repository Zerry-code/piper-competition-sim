#!/usr/bin/env bash
set -e
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -o Acquire::Retries=2 -o Acquire::http::Timeout=20
apt-get install -y libgflags-dev libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev libuvc-dev ros-humble-image-geometry ros-humble-camera-info-manager ros-humble-image-transport ros-humble-image-publisher ros-humble-cv-bridge ros-humble-tf2-eigen ros-humble-tf2-sensor-msgs
workspace="$root_dir/tools/ros-camera-ws"
mkdir -p "$workspace/src"
if [ ! -d "$workspace/src/ros2_astra_camera/.git" ]; then
 git clone --depth 1 https://github.com/orbbec/ros2_astra_camera.git "$workspace/src/ros2_astra_camera"
fi
source /opt/ros/humble/setup.bash
cd "$workspace"
colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
