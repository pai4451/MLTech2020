#!/usr/bin/env bash

train=http://www.amlbook.com/data/zip/features.train
test=http://www.amlbook.com/data/zip/features.test

mkdir data
wget "${train}" -O ./data/features.train.txt
wget "${test}" -O ./data/features.test.txt