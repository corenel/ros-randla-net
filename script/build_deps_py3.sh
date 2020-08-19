#!/usr/bin/env bash

# build and install nearest_neighbors
cd "$(git rev-parse --show-cdup)"|| exit
cd src/utils/nearest_neighbors \
  && python3 setup.py install --home="." \
  && touch __init__.py \
  && touch lib/__init__.py \
  && touch lib/python/__init__.py \
  || exit

# build and install cpp_wrappers
#cd "$(git rev-parse --show-cdup)"|| exit
#cd src/utils/cpp_wrappers/cpp_subsampling \
#  && python setup.py build_ext --inplace \
#  && touch ../__init__.py \
#  && touch __init__.py \
#  || exit
