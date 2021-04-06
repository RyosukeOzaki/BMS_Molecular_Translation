#!/bin/sh

cd ../input/inchi-preprocess-2/
cat train2.a* > train2.pkl
rm train2.a*
cd ../inchi-resnet-lstm-with-attention-starter/
cat resnet34_fold0_best.a* > resnet34_fold0_best.pth
rm resnet34_fold0_best.a*
cd ..
mkdir bms-molecular-translation
pip install --upgrade pip' command
