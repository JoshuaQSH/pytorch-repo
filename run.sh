#!/bin/bash

cat requirement.txt | while read line
do
    PIP_CHECK=$(pip list | grep $line > /dev/null; echo $?)
    if [ $PIP_CHECK -ne 0 ]; then
        echo "Consider install the required lib first "
        exit
    fi;
done
echo "Env Check Success"
# specify --use-cuda to use GPU device
python ./train.py --model convnext_small \
       --dataset ImageNet \
       --batch-size 32 \
       --use-cuda \
       --lr 0.01 \
       --device cuda:0 \
