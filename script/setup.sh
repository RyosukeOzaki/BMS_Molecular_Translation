#!/bin/sh

cd ../input/inchi-preprocess-2/
cat train2* > train2.pkl
cd ../inchi-resnet-lstm-with-attention-starter/
cat resnet34_fold0_best.* > resnet34_fold0_best.pth

