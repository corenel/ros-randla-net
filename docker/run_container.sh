#!/usr/bin/env bash

docker stop ros-randla-net || true && docker rm ros-randla-net || true
docker run -it \
  --name ros-randla-net \
  --privileged \
  --ipc=host \
  --network=host \
  --gpus all \
  --volume="$(dirname ${PWD}):/workspace" \
  ${@} \
  corenel/ros-randla-net
