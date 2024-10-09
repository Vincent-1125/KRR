import argparse
import time
from mpi4py import MPI
from utils import *


def main(args):
    # initialize
    t0 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N, L, S = args.Number_of_Processes, args.Lambda, args.Sigma

    # 1. read and split the data
    if rank == 0: read_and_split(N) 
    comm.Barrier()
    t1 = time.time()

    # 2. compute the K matrix
    compute_k()
    t2 = time.time()

    # 3. calculate the alpha
    solve_linear(Lambda=L)
    t3 = time.time()

    # 4. compute y_pred
    y_pred = predict_y(sigma=S)
    t4 = time.time()

    # 5. calculate the loss and print the results
    if rank == 0:
        loss = compute_loss(y_pred)
        print(f"Lambda: {L} \tSigma: {S} \tNumber of Processes: {N}")
        print(f"RMSE loss: {round(loss, 4)}")
        print(f"""Time comsumption: 
            {round(t1 - t0, 2)}s for reading and splitting data, 
            {round(t2 - t1, 2)}s for computing matrix k, 
            {round(t3 - t2, 2)}s for solving alpha, 
            {round(t4 - t3, 2)}s for predicting y.
            Total time: {round(t4 - t0)}s.""")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--Number_of_Processes', type=int, required=True, help="Number_of_Processes")
    parser.add_argument('-L', '--Lambda', type=float, required=False, default=1, help="Lambda")
    parser.add_argument('-S', '--Sigma', type=float, required=False, default=1, help="Sigma")
    args = parser.parse_args()

    main(args)