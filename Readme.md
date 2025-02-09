
## Project Overview

This project aims to predict the median house value using Kernel Ridge Regression (KRR). The implementation is parallelized using the Message Passing Interface (MPI) to expedite the computation.

The dataset contains the housing price information in California from the 1990 Census. The original data can be found at https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html. In this file, each row corresponds to an instance, and the meanings of the columns are as follows:

1. longitude: A measure of how far west a house is; a higher value is farther west
2. latitude: A measure of how far north a house is; a higher value is farther north
3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
4. totalRooms: Total number of rooms within a block
5. totalBedrooms: Total number of bedrooms within a block
6. population: Total number of people residing within a block
7. households: Total number of households, a group of people residing within a home unit, for a block
8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
9. oceanProximity: Location of the house w.r.t ocean/sea (0 = <1H OCEAN, 1 = INLAND, 2 = NEAR OCEAN, 3 = NEAR BAY)
10. medianHouseValue: Median house value for households within a block (measured in US Dollars)


## Requirements

- Python 3.x
- Numpy
- Pandas
- MPI for Python (mpi4py)

## Installation

Install the required packages using pip:

```bash
pip install numpy pandas mpi4py
```

Download or clone the repository containing the project files.

## Running the Code

1. Make sure you have an MPI environment set up. You can use `mpiexec` or `mpirun` to run the Python script in parallel.
2. Run the main script with the number of processes as an argument. For example, to run with 8 processes:

```bash
mpiexec -n 8 python main.py -N 8 -L 1 -S 1
```
3. For simplicity, you can also just run this:
```
chmod +x main.sh
./main.sh
```

**Parameters**:

**N**: Number of processes(required)

**L**: regularization parameter lambda (optional, default 1)

**S**: Sigma (optional, default 1)

Replace `8` with the desired number of processes.

The execution time is less then 10 seconds on my device, which is faster than the **sklean.kernel_ridge** (taking 15 seconds)!!! A typical output is like this:

```
Lambda: 1.0   Sigma: 1.0    Number of Processes: 8
RMSE loss: 0.4786
Time comsumption:
    0.03s for reading and splitting data,
    5.03s for computing matrix k,
    2.82s for solving alpha,
    1.36s for predicting y.
    Total time: 9s.
```

## Files Description

- `main.py`: The main script that orchestrates the reading, splitting, and processing of the data, as well as the computation of the KRR model.
- `utils.py`: Contains utility functions for standardizing data, computing kernels, matrix operations, solving linear system,  and other helper functions necessary for KRR.
- `/Dataset`: To store the training and test dataset.
- `/output`: To store the middle results like K matrix, alpha, y_pred, etc.
- `main.sh` Run this script to get the whole program started in default parameters.
- `tuning.sh` Run this script to perform grid search and find the optimal parameters.

## Methodology

1. **Data Splitting**: The dataset is split into training (70%) and test (30%) sets, using random seed 42.
2. **Standardization**: Both training and test datasets are standardized to have a mean of zero and a standard deviation of one.
3. **Kernel Ridge Regression**:
   - Gaussian Kernel is used to compute the kernel matrix.
   - Ridge regression is applied with L2 regularization.
   - MPI is utilized to parallelize the computation of the kernel matrix and the optimization process.
4. **Model Training**: The model is trained on the training data.
5. **Prediction**: The model is applied to test data to predict house values.
6. **Evaluation**: The root mean square error (RMSE) is calculated to evaluate the model's performance.

7. **Tuning Parameters**

   - The parameter `sigma` and `lambda` are tuned using gridsearch.

   - candidate_values are: 
   lambdas = [0.01, 0.1, 1, 10]
   sigmas = [0.1, 0.5, 1, 2]

   - The optimal parameters are: lambda = 1, sigma = 1.

## Results

- **Testing RMSE** (optimal): 0.47

## Contact

For any queries or clarifications, please contact:

- E1351295@u.nus.edu Tianyi Chen
