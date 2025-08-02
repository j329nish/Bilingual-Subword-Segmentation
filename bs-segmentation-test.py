from scipy.special import logsumexp
import numpy as np
from tqdm import tqdm
import argparse
import re

# データの入力(alpha)
def set_alpha(input_alpha):
    alpha = []
    with open(input_alpha, "r", encoding="utf-8") as f:
        for line in f:
            alpha.append([float(x) for x in line.strip().split('|||')])
    src_alpha = logsumexp(np.array(alpha, dtype=np.float64), axis=1)
    return src_alpha, np.min(src_alpha)

# データの入力
def set_input(input, input_id2token):
    token2id = {}
    with open(input_id2token, "r", encoding="utf-8") as f:
        for id, line in enumerate(f):
            token = line.strip()
            token2id[token] = id

    text, X, Pu_X = [], [], []
    pattern = re.compile(r"(.*?)\|\|\|(-?\d+(?:\.\d+)?)")
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            text_i, X_i, Pu_X_i = [], [], []
            matches = pattern.findall(line.strip())
            for segment_text, score in matches:
                tokens = segment_text.strip().split()
                text_i.append(tokens)
                X_i.append([token2id.get(token, -1) for token in tokens])
                Pu_X_i.append(np.float64(score))
            text.append(text_i)
            X.append(X_i)
            Pu_X.append(Pu_X_i)
    return text, X, Pu_X

# アライメントの計算
def Pm(X, Pu_X, alpha, MIN):
    N = len(X)
    bestX = []
    for n in tqdm(range(N)):
        max = -np.inf
        best = 0
        for k in range(len(X[n])):
            X_tokens = [u for u in X[n][k] if u != -1]
            submatrix = alpha[X_tokens] if X_tokens else np.array([])
            num_missing = len(X[n][k]) - len(X_tokens)
            prod_alpha = np.sum(np.maximum(submatrix, MIN)) + num_missing * MIN
            current = Pu_X[n][k] + prod_alpha
            if current > max:
                max = current
                best = k
        bestX.append(best)
    return bestX

# サブワード文対の出力
def set_output(text, best, output):
    with open(output, "w", encoding="utf-8") as f:
        for n, best_idx in enumerate(best):
            f.write(" ".join(text[n][best_idx]) + "\n")

def main(args):
    # ファイルの指定
    input = args.input
    input_id2token = args.input_id2token
    input_alpha = args.input_alpha
    output = args.output

    # データの入力
    alpha, MIN = set_alpha(input_alpha)
    textX, X, Pu_X = set_input(input, input_id2token)

    # アライメントの計算
    bestX = Pm(X, Pu_X, alpha, MIN)

    # サブワード文対の出力
    set_output(textX, bestX, output)
    print("finished")

# 引数の指定
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--input_id2token', type=str, default=None)
    parser.add_argument('--input_alpha', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
