#!/bin/bash

docker run -it --ipc="host" --name gyre-container -p 8000:8000 --mount type=bind,source="$(pwd)",target=/home --gpus all gyre /bin/bash



docker run -it --ipc="host" --name gyre-container2 -p 8002:8002 --mount type=bind,source="$(pwd)",target=/home --gpus 1 gyre /bin/bash


docker run -it --ipc="host" --name gyre-container3 -p 8003:8003 --mount type=bind,source="$(pwd)",target=/home --gpus '"device=2"' gyre /bin/bash
