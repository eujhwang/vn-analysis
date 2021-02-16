#!/usr/bin/env bash

if source activate ogb; then
	:
else
	echo 'Creating environment'
	conda create -n ogb python=3.7 -y
	source activate ogb
fi

CUDA=$($PYTHON -c "import torch; print(torch.version.cuda)")

# see instructions at https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=$CUDA -c pytorch

TORCH=$($PYTHON -c "import torch; print(torch.__version__)")
TORCH=${TORCH:0:5}

if [ "${CUDA}" = "None" ]; then
  CUDA="cpu"
else
  CUDA="cu${CUDA/./}"
fi

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric
#pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
#pip install torch-geometric
pip install -r requirements.txt