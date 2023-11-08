mamba create -y --name vod python=3.11
mamba activate vod
mamba install -y -c nvidia cuda cudnn cudatoolkit
mamba install -y -c conda-forge cmake==3.23.1 libblas liblapack mkl mkl-include swig==4.1.1 numpy openblas libgcc
mamba install -y pip poetry
# mamba install -c "pytorch/label/nightly" faiss-gpu
poetry install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # re-install torch to get the right cuda version
bash scripts/build-faiss.sh


# strings /home/$USER/mambaforge/envs/vod/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
# export LD_LIBRARY_PATH=/home/$USER/mambaforge/envs/vod/lib:$LD_LIBRARY_PATH

# Fix GLIBCXX_3.4.30 linking error
# strings /home/$USER/mambaforge/envs/vod/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
# export LD_LIBRARY_PATH=/home/$USER/mambaforge/envs/vod/lib:$LD_LIBRARY_PATH