#!/bin/bash
# map feature using indicator encoding, also produce featmap.txt
python mapfeat.py
# split train and test
python mknfold.py points.txt 1

XGBOOST=../../../xgboost

# training and output the models
$XGBOOST points.conf
# output prediction task=pred
$XGBOOST points.conf task=pred model_in=0002.model
# print the boosters of 00002.model in dump.raw.txt
$XGBOOST points.conf task=dump model_in=0002.model name_dump=dump.raw.txt
# use the feature map in printing for better visualization
$XGBOOST points.conf task=dump model_in=0002.model fmap=pointsmap.txt name_dump=dump.nice.txt
cat dump.nice.txt
