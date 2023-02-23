import copy
import math

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class FeatureDataset(Dataset):
    def __init__(self, file_name):
        file_out = pd.read_excel(file_name)
        x = file_out.iloc[0:16833, 0:10].values
        y = file_out.iloc[0:16833, 10].values

        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


class FeatureDataset1(Dataset):
    def __init__(self, file_name):
        file_out = pd.read_excel(file_name)
        x = file_out.iloc[0:516, 0:10].values
        y = file_out.iloc[0:516, 10].values

        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        cost += (f_wb - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw


def gradient_descent(x, y, w_in, b_in, cost_function, grad_function, alpha, num_iters):
    J_list = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = grad_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_list.append(cost_function(x, y, w, b))
        if i % 100 == 0:
            print(f"Iteration {i} -  Cost: {J_list[-1]} ")
        if i % 50 == 0:
            alpha *= 0.9
    return w, b, J_list


data = FeatureDataset(r"C:\Users\ASUS\Desktop\Data collection\Book1.xlsx")
train = FeatureDataset1(r"C:\Users\ASUS\Desktop\Data collection\Book2.xlsx")
x_1 = data.X_train
y_1 = data.Y_train
initial_w = [0, 0, 0, 0,
             0, 0, 0, 0, 0, 0]
initial_b = 0
iterations = 1000
alpha_1 = 0.01
w_final, b_final, J_hist = gradient_descent(x_1, y_1, initial_w, initial_b,
                                            compute_cost, compute_gradient, alpha_1, iterations)

x_test = train.X_train.numpy()
y_test = train.Y_train.numpy()
m = 500
print(f"Weight: {w_final}")
print(f"Bias: {b_final}")
print(f"Final cost: {J_hist[-1]}")
for i in range(m):
    print(f"Prediction: {np.dot(x_test[i], w_final) + b_final}, Target value: {y_test[i]}")
