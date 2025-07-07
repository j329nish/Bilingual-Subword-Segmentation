#!/bin/bash

mkdir -p ../dataset/top-k
mkdir -p ../dataset/top-k/train
mkdir -p ../dataset/top-k/dev
mkdir -p ../dataset/top-k/test
for lang in src tar; do
    for type in train/train dev/dev test/test; do
        INPUT=../dataset/data/$type.$lang
        MODEL=../dataset/spm-model/spm.$lang.model
        OUT_DIR=../dataset/top-k/$(dirname $type)

        python3 <<EOF
import sentencepiece as spm
import os

sp = spm.SentencePieceProcessor()
sp.load("$MODEL")

input_path = "$INPUT"
out_base = os.path.join("$OUT_DIR", "$(basename $type)")

# 確率の計算
with open(input_path, "r", encoding="utf-8") as f_in, \
     open(f"{out_base}.$lang", "w", encoding="utf-8") as f_out:

    for line in f_in:
        proto = sp.nbest_encode(line.strip(), nbest_size=10, out_type='immutable_proto')
        merged = []
        for nbest in proto.nbests:
            text = " ".join([p.piece for p in nbest.pieces])
            score = nbest.score
            merged.append(f"{text}|||{score}")
        f_out.write("|||".join(merged) + "\n")
print("finished")
EOF

    done
done
