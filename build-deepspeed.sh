CURR_DIR=$(pwd)
PIPCMD=`poetry run which pip`

# Cloning deepspeed
mkdir libs
cd libs
git clone https://github.com/microsoft/DeepSpeed.git
cd deepspeed

# install Cmake, BLAS, MKL and swig
# sudo apt-get install libaio-dev ninja-build
sudo yum install libaio-devel ninja-build

# DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 $PIPCMD install .


# https://github.com/pytorch/pytorch/issues/47717
# use ninja to dynamically build ops
mamba install -c conda-forge gcc_linux-64 gxx_linux-64 -y
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0
poetry run pip install --pre deepspeed --no-cache -v --disable-pip-version-check  2>&1 | tee build.log

# poetry run pip install libs/DeepSpeed --no-cache -v --disable-pip-version-check  2>&1 | tee build.log
# $PIPCMD install . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log

# go back to the root directory
cd $CURR_DIR


# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
