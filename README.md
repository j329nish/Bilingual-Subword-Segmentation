# バイリンガルサブワード分割
## 概要
「アライメントを用いたバイリンガルサブワード分割」の研究に関するコード群です。<br>
以下に示すファイル構成は使用例を示しているだけなので、必要なファイルは各自作成してください。

## データの準備
```
┣ dataset/
    ┣ data/
        ┣ train/  - 学習用データ
            ┣ train.src
            ┣ train.tar
        ┣ dev/    - 検証用データ
            ┣ dev.src
            ┣ dev.tar
        ┣ test/   - 評価用データ
            ┣ test.src
            ┣ test.tar
```

## 環境構築
- Python 3.10.12
- 各種パッケージは`requirements.txt`参照

## フォルダ構成
```
┣ dataset/
    ┣ data/                     - 元のデータセット
    ┣ spm-model/                - SentencePieceモデル用
    ┣ top-k/                    - データのtop-k
    ┣ unigram/                  - サブワード分割されたファイル
    ┣ bs-seg/                   - バイリンガルサブワード分割されたファイル
    ...
┣ scripts/
    ┣ train-spm.sh                  - SentencePieceモデルの訓練
    ┣ top-k.sh                      - top-kの出力
    ┣ em-make-alignment.py          - アライメント行列の作成
    ┣ bs-segmentation.py            - バイリンガルサブワード分割（train, valid）
    ┣ bs-segmentation-viterbi.py    - バイリンガルサブワード分割（train, valid）by viterbi
    ┣ bs-segmentation-test.py       - バイリンガルサブワード分割（test）
┣ alignment/
```

## 実行
### SentencePieceモデルの訓練
```bash
./train-spm.sh
```

### top-kの出力
```bash
./top-k.sh
```

### EMアルゴリズムを用いたアライメント行列の作成
```python
python em-make-alignment.py \
    # --input_alpha ../alignment/alpha-step=0 \        - アライメント行列の途中過程（途中で止まった時用）
    --smoothed_log_value=-1e10 \                       - アライメントlog確率の最小値
    --input_src ../dataset/top-k/train/train.src \     - 原言語の訓練データのtop-k
    --input_tar ../dataset/top-k/train/train.tar \     - 目的言語の訓練データのtop-k
    --output_id2token_src ../alignment/id2token.src \  - 原言語のidとトークン辞書
    --output_id2token_tar ../alignment/id2token.tar \  - 目的言語のidとトークン辞書
    --output_alpha ../alignment/alpha.txt              - アライメント行列
```

### アライメントを用いて対訳文対の作成
```python
# 訓練データ, 検証データ用
python bs-segmentation.py \
    --input_src ../dataset/top-k/train/train.src \    - 原言語の訓練/検証データのtop-k
    --input_tar ../dataset/top-k/train/train.tar \    - 目的言語の訓練/検証データのtop-k
    --input_id2token_src ../alignment/id2token.src \  - 原言語のidとトークン辞書
    --input_id2token_tar ../alignment/id2token.tar \  - 目的言語のidとトークン辞書
    --input_alpha ../alignment/alpha-epoch=0.txt \    - アライメント行列
    --output_src ../dataset/bs-seg/train/train.src \  - 原言語の訓練/検証データの出力ファイル
    --output_tar ../dataset/bs-seg/train/train.tar    - 目的言語の訓練/検証データの出力ファイル

# viterbi版
python bs-segmentation-viterbi.py \
    --input_src ../dataset/top-k/train/train.src \    - 原言語の訓練/検証データのtop-k
    --input_tar ../dataset/top-k/train/train.tar \    - 目的言語の訓練/検証データのtop-k
    --input_id2token_src ../alignment/id2token.src \  - 原言語のidとトークン辞書
    --input_id2token_tar ../alignment/id2token.tar \  - 目的言語のidとトークン辞書
    --input_alpha ../alignment/alpha-epoch=0.txt \    - アライメント行列
    --output_src ../dataset/bs-seg/train/train.src \  - 原言語の訓練/検証データの出力ファイル
    --output_tar ../dataset/bs-seg/train/train.tar    - 目的言語の訓練/検証データの出力ファイル

# 評価データ用
python bs-segmentation-test.py \
    --input ../dataset/top-k/test/test.src \        - 原言語の評価データのtop-k
    --input_id2token ../alignment/id2token.src \    - 原言語のidとトークン辞書
    --input_alpha ../alignment/alpha-epoch=0.txt \  - アライメント行列
    --output ../dataset/bs-seg/train/train.src      - 原言語の評価データの出力ファイル
touch ../dataset/bs-seg/test/test.tar               - 目的言語の評価データの出力ファイル
```

(最終更新 2025/7/14)
