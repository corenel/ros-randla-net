#!/usr/bin/env bash

cp ../requirements.txt .
docker build -t corenel/ros-randla-net -f Dockerfile .
[ -f ./requirements.txt ] && rm -rf ./requirements.txt
