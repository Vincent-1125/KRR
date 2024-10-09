#!/bin/bash

# a simple script to run the main.py script with default parameters

mpiexec -n 8 python main.py -N 8 -L 1 -S 1