#!/usr/bin/env bash

# build and install nearest_neighbors
cd "$(git rev-parse --show-cdup)"|| exit
cd src/utils/nearest_neighbors \
  && python setup.py install --home="." \
  && touch __init__.py \
  && touch lib/__init__.py \
  && touch lib/python/__init__.py \
  || exit

# build and install cpp_wrappers
cd "$(git rev-parse --show-cdup)"|| exit
cd src/utils/cpp_wrappers/cpp_subsampling \
  && python3 setup.py build_ext --inplace \
  && touch ../__init__.py \
  && touch __init__.py \
  && touch grid_subsampling/__init__.py \
  || exit
