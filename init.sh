# download the contents of the dramsim3 submodule
# git submodule update --init --recursive

# cd dramsim3
# mkdir build
# cd build
# cmake ..

# make -j4

# cd ../..
# echo dramsim installation is done


export PATH="$PATH:$(pwd)/gem5/ext/dramsim3/DRAMsim3/build"

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Initializing main repository submodules (Level 1)..."
# Initialize and clone the first layer (X1) without cloning its nested submodules yet
git submodule update --init

echo "Entering submodule gem5..."
# Navigate to gem5
cd gem5

echo "Setting up branch gem5_ta for gem5..."
# Fetch all branches, checkout to gem5_ta, and pull latest
git fetch origin
git checkout gem5_ta
git pull origin gem5_ta

echo "Initializing nested submodules within gem5 (Level 2)..."
# Now initialize DRAMsim3 which lives inside gem5
git submodule update --init

echo "Entering nested submodule DRAMsim3..."
# Navigate to DRAMsim3 (replace 'path/to/DRAMsim3' with the actual folder path inside gem5)
cd ext/dramsim3/DRAMsim3

echo "Setting up branch dramsim_ta for DRAMsim3..."
# Fetch, checkout to dramsim_ta, and pull latest for the nested repo
git fetch origin
git checkout dramsim_ta
git pull origin dramsim_ta

echo "All submodules initialized, set to correct branches, and updated!"