#!/bin/sh
uv sync
uv run mnist-helper.py
python main.py MNIST.keras mnist 
