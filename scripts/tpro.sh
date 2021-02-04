#!/bin/bash

set -e

source activate ogb

# set paths to an external disk or similar if you want the saved checkpoints / datasets
# to be stored somewhere else different from the repo folder
SAVE=$PWD
DATA=""

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PROJECT

echo "PYTHONPATH: ${PYTHONPATH}"
echo " Run started at:- "
date

PROJECT=$PWD
SAVE=$SAVE/s_tpro/
RESULTS=$PROJECT/r_tpro/

MODEL=sage
HIDDEN=256
LAYERS=(3)
LRS=(5e-3 1e-2 5e-2 1e-2 1e-3 5e-4 1e-4)

## for tuning the models we only select a subset of the data and folds
TIDX=train50  # set "" to run on full dataset
FOLDS=5       # set 10 for final experiments
EPOCHS=1000   # set 2000 for final experiments
PAT=100

cd $PROJECT/src

for LAYER in ${LAYERS[*]}; do
  for LR in ${LRS[*]}; do

        NAME="${MODEL}_l${LAYER}_lr${LR}"
        if [[ "$TIDX" != "" ]]; then
            NAME="${NAME}_t${TIDX: -2}"
        fi

        CHECKPOINT=""
        # check if there is a checkpoint for this configuration
        for f in $SAVE/$NAME*; do
            ## Check if the glob gets expanded to existing files.
            ## If not, f here will be exactly the pattern above
            ## and the exists test will evaluate to false.
            ## [ -e "$f" ] && echo "files do exist $f" || echo "files do not exist"
            if [ -e "$f" ]; then
              CHECKPOINT=`basename $f`
              echo "checkpoint exists! $f"
              echo "using: $CHECKPOINT"
              break
            fi
            ## This is all we needed to know, so we can break after found it

        done

        python main_pro.py --gnn=$MODEL --layers=$LAYER --hid_dim=$HIDDEN --dropout=0.0 \
                    --lr=$LR --runs=$FOLDS --epochs=$EPOCHS --patience=$PAT \
                    --dir_data=$DATA  --dir_save=$SAVE --dir_results=$RESULTS --filename=$NAME  \
                    --checkpointing=0 --checkpoint=$CHECKPOINT --train_idx=$TIDX

  done
done



echo "Run completed at:- "
date
