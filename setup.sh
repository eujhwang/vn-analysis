#!/usr/bin/env bash

CUDA_TK=10.1

if source activate ogb; then
	:
else
	echo 'Creating environment'
	conda create -n ogb python=3.7 -y
	source activate ogb
fi

# see instructions at https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=$CUDA_TK -c pytorch

TORCH=$($PYTHON -c "import torch; print(torch.__version__)")
TORCH=${TORCH:0:5}

CUDA=$($PYTHON -c "import torch; print(torch.version.cuda)")
if [ "${CUDA}" = "None" ]; then
  CUDA="cpu"
else
  CUDA="cu${CUDA/./}"
fi

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
pip install -r requirements.txt