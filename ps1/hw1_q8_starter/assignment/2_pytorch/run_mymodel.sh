#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --lr .01 --momentum 0.9 \
    --weight-decay .001 --batch-size 64 \
    --epochs 75 --model mymodel \
    --hidden-dim 32 --kernel-size 5 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
