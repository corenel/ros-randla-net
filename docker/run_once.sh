#!/usr/bin/env bash

docker run -it --rm \
  --privileged \
  --ipc=host \
  --network=host \
  --gpus all \
  --volume="$(dirname ${PWD}):/workspace" \
  ${@} \
  corenel/ros-randla-net