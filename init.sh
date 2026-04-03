# download the contents of the dramsim3 submodule
git submodule update --init --recursive

cd dramsim3
mkdir build
cd build
cmake ..

make -j4

cd ../..
echo dramsim installation is done
