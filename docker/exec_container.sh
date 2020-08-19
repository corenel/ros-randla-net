#!/usr/bin/env bash

docker start ros-randla-net
docker exec -it ros-randla-net /bin/bash /entrypoint.sh /bin/zsh
