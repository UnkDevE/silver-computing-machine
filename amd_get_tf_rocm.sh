#!/bin/bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/tensorflow_rocm-2.16.1-cp39-cp39-manylinux_2_28_x86_64.whl
pip install --user tensorflow_rocm-2.16.1-cp39-cp39-manylinux_2_28_x86_64.whl --upgrade