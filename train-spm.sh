#!/bin/bash

mkdir -p ../dataset/unigram
mkdir -p ../dataset/unigram/train
mkdir -p ../dataset/unigram/dev
mkdir -p ../dataset/unigram/test
mkdir -p ../dataset/spm-model

# sentencepiece model の学習
spm_train --input=../dataset/data/train/train.src --model_prefix=../dataset/spm-model/spm.src --vocab_size=16000 --character_coverage=1.0 --model_type=unigram
spm_train --input=../dataset/data/train/train.tar --model_prefix=../dataset/spm-model/spm.tar --vocab_size=16000 --character_coverage=1.0 --model_type=unigram

for lang in src tar
do
    for type in train/train dev/dev test/test
    do
    spm_encode --model=../dataset/spm-model/spm.$lang.model --output_format=piece < ../dataset/data/$type.$lang > ../dataset/unigram/$type.$lang
    done
done
