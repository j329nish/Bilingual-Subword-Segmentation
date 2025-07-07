import numpy as np
from tqdm import tqdm
import argparse

# データの入力(alpha)
def set_alpha(input_alpha):
    alpha = []
    with open(input_alpha, "r", encoding="utf-8") as f:
        for line in f:
            alpha.append([float(x) for x in line.strip().split('|||')])
    alpha = np.array(alpha, dtype=np.float64)
    return alpha, np.min(alpha)

# データの入力
def set_input(input, input_id2token):
    token2id = {}
    with open(input_id2token, "r", encoding="utf-8") as f:
        for id, line in enumerate(f):
            token = line.strip()
            token2id[token] = id

    text, X, Pu_X = [], [], []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            text_i, X_i, Pu_X_i = [], [], []
            items = line.strip().split('|||')
            for i in range(0, len(items), 2):
                text_i.append(items[i].split())
                X_i.append([token2id.get(token, -1) for token in items[i].split()])
                Pu_X_i.append(np.float64(items[i+1]))
            text.append(text_i)
            X.append(X_i)
            Pu_X.append(Pu_X_i)
    return text, X, Pu_X

# アライメントの計算
def Pm(X, Pu_X, Y, Pu_Y, alpha, MIN):
    N = len(X)
    bestX, bestY = [], []
    for n in tqdm(range(N)):
        best = [-np.inf, 0, 0]
        for k in range(len(X[n])):
            for l in range(len(Y[n])):
                X_tokens = [u for u in X[n][k] if u != -1]
                Y_tokens = [v for v in Y[n][l] if v != -1]
                submatrix = alpha[np.ix_(X_tokens, Y_tokens)] if X_tokens and Y_tokens else np.array([])
                num_missing = len(X[n][k]) * len(Y[n][l]) - len(X_tokens) * len(Y_tokens)
                prod_alpha = np.sum(np.maximum(submatrix, MIN)) + num_missing * MIN
                current = Pu_X[n][k] + Pu_Y[n][l] + prod_alpha
                if current > best[0]:
                    best = [current, k, l]
        bestX.append(best[1])
        bestY.append(best[2])
    return bestX, bestY

# サブワード文対の出力
def set_output(text, best, output):
    with open(output, "w", encoding="utf-8") as f:
        for n, best_idx in enumerate(best):
            f.write(" ".join(text[n][best_idx]) + "\n")

def main(args):
    # ファイルの指定
    input_src = args.input_src
    input_tar = args.input_tar
    input_id2token_src = args.input_id2token_src
    input_id2token_tar = args.input_id2token_tar
    input_alpha = args.input_alpha
    output_src = args.output_src
    output_tar = args.output_tar

    # データの入力
    alpha, MIN = set_alpha(input_alpha)
    textX, X, Pu_X = set_input(input_src, input_id2token_src)
    textY, Y, Pu_Y = set_input(input_tar, input_id2token_tar)

    # アライメントの計算
    bestX, bestY = Pm(X, Pu_X, Y, Pu_Y, alpha, MIN)

    # サブワード文対の出力
    set_output(textX, bestX, output_src)
    set_output(textY, bestY, output_tar)
    print("finished")

# 引数の指定
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', type=str, default=None)
    parser.add_argument('--input_tar', type=str, default=None)
    parser.add_argument('--input_id2token_src', type=str, default=None)
    parser.add_argument('--input_id2token_tar', type=str, default=None)
    parser.add_argument('--input_alpha', type=str, default=None)
    parser.add_argument('--output_src', type=str, default=None)
    parser.add_argument('--output_tar', type=str, default=None)
    args = parser.parse_args()
    main(args)
