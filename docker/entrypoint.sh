#!/usr/bin/env bash
set -e

# setup ssh
# /usr/sbin/sshd -D &

# setup ros environment
# source "/opt/ros/$ROS_DISTRO/setup.bash"
exec "$@"
