CURR_DIR=$(pwd)
PYPATH=`which python`
echo "Installing faiss for $PYPATH"
pip install swig

# Cloning faiss
mkdir libs
cd libs
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# install Cmake, BLAS, MKL and swig
mamba install -y -c conda-forge cmake==3.23.1 libblas liblapack mkl mkl-include swig==4.1.1 numpy
# mamba install -y -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft

# Build faiss `https://github.com/facebookresearch/faiss/blob/main/INSTALL.md`
# check your cuda arch number at `https://developer.nvidia.com/cuda-gpus`
# tips: use -jY to use Y cores
rm -rf build/
cmake \
-DFAISS_ENABLE_RAFT=OFF \
-DFAISS_OPT_LEVEL=avx2 \
-DFAISS_ENABLE_GPU=ON \
-DFAISS_ENABLE_PYTHON=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_ARCHITECTURES="80;86" \
-DPython_EXECUTABLE=$PYPATH \
-DCMAKE_INSTALL_PREFIX=$PYPATH \
-B build .
make -C build -j16 faiss faiss_avx2
make -C build -j16 swigfaiss swigfaiss_avx2
(cd build/faiss/python && $PYPATH setup.py install)

# Optional: install C headers
make -C build install

# go back to the root directory
cd $CURR_DIR



