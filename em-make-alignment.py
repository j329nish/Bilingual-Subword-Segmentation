from scipy.special import logsumexp
import numpy as np
from tqdm import tqdm
import argparse
import os
from numba import jit
from numba.typed import List
import re

# データの入力(alpha)
def set_input_alpha(input_alpha):
    alpha = []
    with open(input_alpha, "r", encoding="utf-8") as f:
        for line in f:
            alpha.append([float(x) for x in line.strip().split('|||')])
    return np.array(alpha, dtype=np.float64)

# データの入力(input)
def set_input(input, output_id2token):
    X, Pu_X = [], []
    token2id, id2token = {}, {}
    id = 0
    # index = 0
    pattern = re.compile(r"(.*?)\|\|\|(-?\d+(?:\.\d+)?)")
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            X_i, Pu_X_i = [], []
            matches = pattern.findall(line.strip())
            for text, score in matches:
                tokens = []
                for token in text.strip().split():
                    if token not in token2id:
                        token2id[token] = id
                        id2token[id] = token
                        id += 1
                    tokens.append(token2id[token])
                X_i.append(tokens)
                Pu_X_i.append(np.float64(score))
            X.append(X_i)
            Pu_X.append(Pu_X_i)
            # if index == 10:
            #     break
            # index += 1

    with open(output_id2token, "w", encoding="utf-8") as f:
        for _, token in id2token.items():
            f.write(token + "\n")
    return X, Pu_X, len(token2id)

# Mステップ
@jit(nopython=True)
def M_step(lenK, lenL, X_n, Y_n, numerators, denominators_n, alpha_new):
    for k in range(lenK):
        for l in range(lenL):
            E_nkluv = numerators[k, l] - denominators_n
            for u in X_n[k]:
                for v in Y_n[l]:
                    alpha_new[u, v] = np.logaddexp(alpha_new[u, v], E_nkluv)
    return alpha_new

# EMアルゴリズム
def EM_algorithm(N, K, L, X, Y, Pu_X, Pu_Y, smoothed_log_value, alpha_old):
    alpha_new = np.full_like(alpha_old, -np.inf, dtype=np.float64)
    total_log_likelihood = 0.0
    for n in tqdm(range(N)):
        # Eステップ
        numerators = np.full((K, L), -np.inf)
        lenK = len(X[n])
        lenL = len(Y[n])
        for k in range(lenK):
            for l in range(lenL):
                prod_alpha = 0.0
                submatrix = alpha_old[np.ix_(X[n][k], Y[n][l])]
                prod_alpha = np.sum(logsumexp(submatrix, axis=1))
                # E_nkluvの分子, 分母の計算
                numerators[k][l] = Pu_X[n][k] + Pu_Y[n][l] + prod_alpha
        denominators_n = logsumexp(numerators.ravel())

        # 尤度の計算
        if not np.isinf(denominators_n):
            total_log_likelihood += denominators_n

        # numba用配列
        X_n = List()
        for seg in X[n]:
            X_n.append(List(seg))
        Y_n = List()
        for seg in Y[n]:
            Y_n.append(List(seg))

        # Mステップ
        alpha_new = M_step(lenK, lenL, X_n, Y_n, numerators, denominators_n, alpha_new)

    # alpha_newの更新
    finite_mask = np.isfinite(alpha_new)
    if smoothed_log_value == 0:
        smoothed_log_value = np.min(alpha_new[finite_mask]) + np.log(0.5)
    alpha_new[~finite_mask] = smoothed_log_value
    alpha_new -= logsumexp(alpha_new.ravel())
    return alpha_new, total_log_likelihood

# alphaの更新回数
def train(N, K, L, U, V, X, Y, Pu_X, Pu_Y, smoothed_log_value, alpha, output_alpha):
    # alphaの初期値, 適切に設定する必要がある(0より大きくないとダメ)
    if alpha is None:
        alpha = np.log(np.ones([U, V], dtype=np.float64) / (U * V))
    p_old = -np.inf
    epoch = 0
    while (True):
        alpha, p_new  = EM_algorithm(N, K, L, X, Y, Pu_X, Pu_Y, smoothed_log_value, alpha)
        print(f'[epoch {epoch}] P_old(X, Y) = {p_new} min(alpha) = {np.min(alpha)}')
        if (p_new - p_old <= 0):
            break
        set_alpha(alpha, output_alpha, U, epoch)
        p_old = p_new
        epoch += 1

# alphaファイルの出力
def set_alpha(alpha, output_alpha, U, epoch):
    if (epoch != -1):
        filename, ext = os.path.splitext(output_alpha)
        output_alpha = f"{filename}_epoch-{str(epoch)}{ext}"
    with open(output_alpha, "w", encoding="utf-8") as f:
        for u in range(U):
            f.write("|||".join(f'{item}' for item in alpha[u]) + "\n")

def main(args):
    # ファイルの指定
    input_alpha = args.input_alpha
    smoothed_log_value = args.smoothed_log_value
    input_src = args.input_src
    input_tar = args.input_tar
    output_id2token_src = args.output_id2token_src
    output_id2token_tar = args.output_id2token_tar
    output_alpha = args.output_alpha

    # データの入力, id2tokenファイルの出力
    alpha = set_input_alpha(input_alpha) if input_alpha else None
    X, Pu_X, U = set_input(input_src, output_id2token_src)
    Y, Pu_Y, V = set_input(input_tar, output_id2token_tar)
    N, K, L, = len(Pu_X), len(Pu_X[0]), len(Pu_Y[0])

    # 入力データの確認
    for i in range(10):
        for j in range(3):
            print(f"X[{i}][{j}]={X[i][j]}, Pu_X[{i}][{j}]={Pu_X[i][j]}")
            print(f"Y[{i}][{j}]={Y[i][j]}, Pu_Y[{i}][{j}]={Pu_Y[i][j]}")
        print(f"len(X[{i}])={len(X[i])}, len(Pu_X[{i}])={len(Pu_X[i])}")
        print(f"len(Y[{i}])={len(Y[i])}, len(Pu_Y[{i}])={len(Pu_Y[i])}")
    print(f"N={N}, K={K}, L={L}, U={U}, V={V}")

    # alphaの確率を計算
    train(N, K, L, U, V, X, Y, Pu_X, Pu_Y, smoothed_log_value, alpha, output_alpha)
    print("finished")

# 引数の指定
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_alpha', type=str, default=None)
    parser.add_argument('--smoothed_log_value', type=np.float64, default=0)
    parser.add_argument('--input_src', type=str, default=None)
    parser.add_argument('--input_tar', type=str, default=None)
    parser.add_argument('--output_id2token_src', type=str, default=None)
    parser.add_argument('--output_id2token_tar', type=str, default=None)
    parser.add_argument('--output_alpha', type=str, default=None)
    args = parser.parse_args()
    main(args)
