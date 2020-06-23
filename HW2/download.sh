#!/usr/bin/env bash

train=http://amlbook.com/data/zip/zip.train
test=http://amlbook.com/data/zip/zip.test

mkdir data
wget "${train}" -O ./data/zip.train.txt
wget "${test}" -O ./data/zip.test.txt