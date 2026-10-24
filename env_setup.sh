#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

# Necessary for capturing video (resolve vulkan:No DRI3 support)
# https://github.com/isaac-sim/IsaacGymEnvs/issues/126
export MESA_VK_DEVICE_SELECT="10de:2204"