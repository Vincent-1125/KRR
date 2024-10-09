from mpi4py import MPI
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

paths = {}
paths['X_train'] = 'Dataset/X_train.npz'
paths['Y_train'] = 'Dataset/Y_train.npz'
paths['X_test'] = 'Dataset/X_test.npz'
paths['Y_test'] = 'Dataset/Y_test.npz'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def time_step(start, label):
    """record the time and print the time difference"""
    current = time.time()
    print(f"{label} took {round(current - start, 2)} seconds.")
    return current


def get_array(path):
    array = np.load(path, mmap_mode='r')
    array = np.concatenate([array[f'sub_{i}'] for i in range(size)])
    return array


def standardize(X, y):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y


def read_and_split(N):
    # get the trainging data and test data
    train = pd.read_csv('Dataset/housing_train.csv').values
    test = pd.read_csv('Dataset/housing_test.csv').values
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    # standardize the data
    X_train, Y_train = standardize(X_train, y_train)
    X_test, Y_test = standardize(X_test, y_test)

    # split the data
    X_train_split = np.array_split(X_train, N)
    Y_train_split = np.array_split(Y_train, N)
    X_test_split = np.array_split(X_test, N)
    Y_test_split = np.array_split(Y_test, N)

    # store the data into npz files
    X_train_split_map = {}
    Y_train_split_map = {}
    X_test_split_map = {}
    Y_test_split_map = {}

    for i in range(N):
        X_train_split_map[f'sub_{i}'] = X_train_split[i]
        Y_train_split_map[f'sub_{i}'] = Y_train_split[i]
        X_test_split_map[f'sub_{i}'] = X_test_split[i]
        Y_test_split_map[f'sub_{i}'] = Y_test_split[i]

    np.savez(f'Dataset/X_train.npz', **X_train_split_map)
    np.savez(f'Dataset/Y_train.npz', **Y_train_split_map)
    np.savez(f'Dataset/X_test.npz', **X_test_split_map)
    np.savez(f'Dataset/Y_test.npz', **Y_test_split_map)

    if not os.path.exists("output"): os.makedirs("output", exist_ok=True)


def norm2(x1, x2):
    return np.linalg.norm(x1 - x2)


def gaussian_kernel(x1, x2, sigma=1):
    return np.exp(-norm2(x1, x2) ** 2 / (2 * sigma ** 2))


def compute_local_k(local_matrix, foreign_matrix, sigma=1.0):
    # calculate the difference tensor of shape (m, n, d), where d is the feature dimension
    diff = local_matrix[:, np.newaxis, :] - foreign_matrix[np.newaxis, :, :]
    # calculate the square of the L2 norm of the difference tensor
    dist_sq = np.sum(diff ** 2, axis=-1)
    # apply the Gaussian kernel formula
    local_K = np.exp(-dist_sq / (2 * sigma**2))
    return local_K


def compute_k():
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix = np.load(paths['X_train'], mmap_mode='r')
    # read the shapes of sub-matrices and broadcast to all processes
    if rank == 0:
        process_map = {}
        N = 0
        for i in range(size):
            process_map[i] = matrix[f'sub_{i}'].shape[0]
            N += matrix[f'sub_{i}'].shape[0]
    else:
        process_map = None
        N = None
    
    process_map = comm.bcast(process_map, root=0)
    N = comm.bcast(N, root=0)

    # create local_k matrix
    local_K = {f'col_{i}': np.zeros((matrix[f'sub_{rank}'].shape[0], process_map[i])) for i in range(size)}

    # each process read its own sub-matrix
    local_matrix = matrix[f'sub_{rank}']

    # send local_matrix to all processes and receive matrix from all processes
    for round in range(size):   # loop over all rounds
        dest = (rank - round + size) % size
        source = (round + rank) % size
        if round == 0: # no need to send to itself, just compute the local_K
            local_K[f'col_{source}'] = compute_local_k(local_matrix, local_matrix)

        else:
            foreign_matrix = np.empty((process_map[source], local_matrix.shape[1]), dtype=local_matrix.dtype)
            comm.Sendrecv(local_matrix, dest=dest, recvbuf=foreign_matrix, source=source)
            local_K[f'col_{source}'] = compute_local_k(local_matrix, foreign_matrix)

    # concatnate all the local_K matrices
    local_K_cat = np.hstack([local_K[f'col_{i}'] for i in range(size)])

    # save the local_K matrix
    if not os.path.exists('output/K'): os.makedirs('output/K', exist_ok=True)
    np.save(f'output/K/K_{rank}.npy', local_K_cat)


def mv_mul(A, x):
    m, n = A.shape
    v = np.zeros(n, dtype=np.double)
    comm.Allgatherv(x, v)
    return A @ v


def inner_product(a, b):
    local_sum = np.dot(a, b)
    result = np.zeros(1, dtype=np.double)
    comm.Allreduce(local_sum, result, op=MPI.SUM)
    return result[0]


def conjugate_gradient(A, y, alpha, threshold=1e-1):
    loss = []
    r = y - mv_mul(A, alpha)
    p = r.copy()
    SE = inner_product(r, r)
    
    while SE > threshold:
        w = mv_mul(A, p)
        s = SE / (inner_product(p, w) + 1e-10)
        alpha = alpha + s * p
        r = r - s * w
        newSE = inner_product(r, r)
        beta = newSE / SE
        p = r + beta * p
        SE = newSE
        loss.append(np.log(SE))
    
    return alpha, loss


def get_A_y(Lambda, rank, size):

    train_size = 14303
    sub_train_size = train_size // size + 1
    Index = [0 + sub_train_size * i for i in range(size)]

    sub_k = np.load(f"output/K/K_{rank}.npy")
    sub_y = np.load(f"Dataset/Y_train.npz", mmap_mode='r')[f'sub_{rank}']

    # add Lambda to the diagonal of sub_k
    Lambda = 1
    m, n = sub_k.shape
    for i in range(m):
        j = Index[rank] + i
        sub_k[i, j] += Lambda
    
    return sub_k, sub_y


def solve_linear(Lambda):
    rank = comm.Get_rank()
    size = comm.Get_size()
    matrix, y = get_A_y(Lambda, rank, size)

    alpha = np.ones(matrix.shape[0], dtype=np.double)
    alpha, loss = conjugate_gradient(matrix, y, alpha)

    gathered_alpha = None
    if rank == 0:
        gathered_alpha = np.empty((size, alpha.size), dtype=np.double)

    comm.Gather(alpha, gathered_alpha, root=0)

    if not os.path.exists("output/alpha"): os.makedirs("output/alpha", exist_ok=True)

    np.save(f"output/alpha/alpha_{rank}.npy", alpha)

    if rank == 0:
        np.save("output/alpha/gathered_alpha.npy", gathered_alpha)
        plt.plot(loss)
        plt.savefig("output/loss.png")


def predict_y(sigma):
    alpha = np.load(f"output/alpha/alpha_{rank}.npy")
    X_train_local = np.load(paths['X_train'], mmap_mode='r')[f'sub_{rank}']
    X_test = get_array(paths['X_test'])
    y_pred_local = np.zeros(X_test.shape[0])
    y_pred = np.zeros(X_test.shape[0])

    chunk_size = 500

    for start in range(0, X_train_local.shape[0], chunk_size):
        end = min(start + chunk_size, X_train_local.shape[0])
        # calculate the diff matrix of the current chunk: shape (chunk_size, n, d)
        diff_chunk = X_train_local[start:end, np.newaxis, :] - X_test[np.newaxis, :, :]
        gk_chunk = np.exp(-np.sum(diff_chunk**2, axis=2) / (2 * sigma**2))
        y_pred_local_chunk = np.dot(gk_chunk.T, alpha[start:end])
        y_pred_local += y_pred_local_chunk

    comm.Reduce(y_pred_local, y_pred, op=MPI.SUM, root=0)
    return y_pred if rank == 0 else None


def compute_loss(y_pred):
    y_test = get_array(paths['Y_test'])
    loss = np.sqrt(np.sum((y_pred - y_test) ** 2) / y_test.size)
    np.save('output/y_test', y_test)
    np.save('output/y_pred', y_pred)
    return loss


lambdas = [0.01, 0.1, 1, 10]
sigmas = [0.1, 0.5, 1, 2]