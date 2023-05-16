mamba create --name faiss python=3.10
mamba activate faiss
# mamba install -c "pytorch/label/nightly" faiss-gpu>=1.7.3
mamba install -c conda-forge cudnn cudatoolkit
pip install -r requirements.txt
pip install torch