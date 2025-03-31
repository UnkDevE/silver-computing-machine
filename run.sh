#!/bin/sh
uv sync
uv run mnist-helper.py
uv run main.py MNIST.keras mnist 
