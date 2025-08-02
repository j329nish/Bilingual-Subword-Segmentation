import numpy as np
from tqdm import tqdm
import argparse
from numba import jit
from numba.typed import List
import re

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

@jit(nopython=True)
def logsumexp(log_probs):
    a_max = np.max(log_probs)
    if a_max == -np.inf:
        return -np.inf
    s = 0.0
    for x in log_probs:
        s += np.exp(x - a_max)
    return a_max + np.log(s)

# 最適な文対の取得
@jit(nopython=True)
def Pm_single(X_n, Pu_X_n, Y_n, Pu_Y_n, alpha, MIN):
    best_score = -np.inf
    best_k = 0
    best_l = 0
    for k in range(len(X_n)):
        for l in range(len(Y_n)):
            X_tokens = [u for u in X_n[k] if u != -1]
            Y_tokens = [v for v in Y_n[l] if v != -1]
            if X_tokens and Y_tokens:
                prod_alpha = 0.0
                for u in X_tokens:
                    log_probs = np.empty(len(Y_tokens), dtype=np.float64)
                    for i in range(len(Y_tokens)):
                        log_probs[i] = alpha[u, Y_tokens[i]]
                    prod_alpha += logsumexp(log_probs)
            else:
                prod_alpha = len(X_n[k]) * MIN
            current = Pu_X_n[k] + Pu_Y_n[l] + prod_alpha
            if current > best_score:
                best_score = current
                best_k = k
                best_l = l
    return best_k, best_l

# アライメントの計算
def Pm(X, Pu_X, Y, Pu_Y, alpha, MIN):
    N = len(X)
    bestX, bestY = [], []
    for n in tqdm(range(N)):
        # numba用配列
        X_n = List()
        for seg in X[n]:
            X_n.append(List(seg))
        Y_n = List()
        for seg in Y[n]:
            Y_n.append(List(seg))

        # 最適な文対の取得
        best_k, best_l = Pm_single(X_n, Pu_X[n], Y_n, Pu_Y[n], alpha, MIN)
        bestX.append(best_k)
        bestY.append(best_l)
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
