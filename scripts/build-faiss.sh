CURR_DIR=$(pwd)
PYPATH=`which python`
echo "Installing faiss for $PYPATH"
pip install swig

# sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# sudo apt-get update
# sudo apt-get install gcc-4.9
# sudo apt-get upgrade libstdc++6

# Cloning faiss
mkdir libs
cd libs
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# install Cmake, BLAS, MKL and swig
mamba install -y -c conda-forge cmake==3.23.1 libblas liblapack mkl mkl-include swig==4.1.1 numpy openblas libgcc
# mamba install -c conda-forge gcc=12.1.0
# mamba install -y -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft

# Build faiss `https://github.com/facebookresearch/faiss/blob/main/INSTALL.md`
# check your cuda arch number at `https://developer.nvidia.com/cuda-gpus` or `https://en.wikipedia.org/wiki/CUDA`
# tips: use -jY to use Y cores
rm -rf build/
cmake \
-DFAISS_ENABLE_RAFT=OFF \
-DFAISS_OPT_LEVEL=avx2 \
-DFAISS_ENABLE_GPU=ON \
-DFAISS_ENABLE_PYTHON=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_ARCHITECTURES="60;80;86;87;90" \
-DPython_EXECUTABLE=$PYPATH \
-DCMAKE_INSTALL_PREFIX=$PYPATH \
-B build .
make -C build -j12 faiss faiss_avx2
make -C build -j12 swigfaiss swigfaiss_avx2
(cd build/faiss/python && $PYPATH setup.py install)

# Optional: install C headers
make -C build install

# go back to the root directory
cd $CURR_DIR

# Install faiss-gpu in your poetry env:
# > export PYPATH=`poetry run which python`
# > (cd build/faiss/python && $PYPATH setup.py install)

# Re-install torch on top:
# > poetry run pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Solve missing GLIBCXX_3.4.30 (append to your source file):
# strings /home/vlievin/mambaforge/envs/vod/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
# export LD_LIBRARY_PATH=/home/vlievin/mambaforge/envs/vod/lib:$LD_LIBRARY_PATH

