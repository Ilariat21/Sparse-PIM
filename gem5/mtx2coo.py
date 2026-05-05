#!/usr/bin/env python

'''
SparsePIM Accelerator functional simulator / trace generator
                      Initially written by (uegook 11/05/2024)
This version of the simulator operates with the FP16 inputs and returns FP16 outputs
The data placement assumes interlea'''


import math
import numpy as np
import sys, os, argparse, scipy.io, copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from scipy.sparse import csc_matrix

def compress_array_to_csc(array):
    vals = []
    rows = []
    cols = [0]
    
    num_rows, num_cols = array.shape

    # Traverse each column
    for c in range(num_cols):
        count = 0  # Track non-zero elements in the current column
        for r in range(num_rows):
            if array[r, c] != 0:  # Check for non-zero entries
                vals.append(array[r, c])
                rows.append(r)
                count += 1
        # Update column pointer after processing each column
        cols.append(cols[-1] + count)
    
    return np.array(vals), np.array(rows), np.array(cols)

def compress_array_to_csr(array):
    vals = []
    cols = []
    rows = [0]
    
    num_rows, num_cols = array.shape

    # Traverse each row
    for r in range(num_rows):
        count = 0  # Track non-zero elements in the current row
        for c in range(num_cols):
            if array[r, c] != 0:  # Check for non-zero entries
                vals.append(array[r, c])
                cols.append(c)
                count += 1
        # Update row pointer after processing each row
        rows.append(rows[-1] + count)
    
    return np.array(vals), np.array(cols), np.array(rows)

# def read_mat(mat, flag):
#     data = loadmat(mat)
#     if 'Problem' in data.keys():
#         problem = data['Problem']
#         contents = problem[0, 0]
#         (_, matrix, _, _, _, _, name, _, _) = contents
#         if isinstance(matrix, csc_array):
#             dense_matrix = matrix.toarray()
#             if flag:
#                 plt.spy(dense_matrix, markersize=1)
#                 plt.show()
#                 plt.savefig(f"matrix_{name[0]}.png")
#             return dense_matrix
#         else:
#             print("Not csc array!")
#     else:
#         print("No 'Problem' key found")

def read_mat(mat, flag):
    data = loadmat(mat, struct_as_record=False, squeeze_me=True)

    if 'Problem' in data:
        problem = data['Problem']

        # Get the sparse matrix and name
        matrix = problem.A  # or problem.__dict__['A']
        name = problem.name
        if "/" in name:
            name = name.split("/")[-1]
        if isinstance(matrix, csc_matrix):
            dense_matrix = matrix.toarray()
            if flag:
                plt.spy(dense_matrix, markersize=0.1)
                plt.savefig(f"matrix_{name}.png")
                # plt.show()
                plt.clf()
            return dense_matrix
        else:
            print("Not a csc_matrix!")
    else:
        print("No 'Problem' key found")

def read_mtx(direc):
    f = open(direc, 'r')
    lines = f.readlines()
    first = 0
    mtx = np.zeros((100, 100), dtype = float)
    M, N, NNZ = 0, 0, 0
    for i in range(len(lines)):
        if lines[i][0] == "%":
            continue
        elif first == 0 and lines[i][0] != "%":
            first = 1
            token = lines[i].split()
            # print("this token", token)
            if len(token)!=3:
                break
            M, N, NNZ = int(token[0]), int(token[1]), int(token[2])
            mtx = np.zeros((M, N), dtype=float)
        else:
            token = lines[i].split()
            if len(token)<2:
                break
            elif len(token)==2:
                r, c, v = int(token[0]), int(token[1]), 1 
            # print(token, i)
            else:
                r, c, v = int(token[0]), int(token[1]), float(token[2])
            mtx[r-1, c-1] = v
    return mtx


# Custom function to convert string to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Argument parsing
parser = argparse.ArgumentParser(description='Generate a random sparse matrix and save to files.')

parser.add_argument('-mat', '--mat', type=str, default='nasa2910.mat', help='Input matrix file')

args = parser.parse_args()

filename = args.mat

if str(filename)[-3:] == 'mat':
    MKarr = read_mat(str(filename), False)
elif str(filename)[-3:] == 'mtx':
    MKarr = read_mtx(str(filename))

nnz = np.count_nonzero(MKarr)

filename = "{}_coo.txt".format(filename.split('/')[-1].split(".")[-2])
print("here you are", filename)

f = open(filename, 'w')
for i in range(MKarr.shape[0]):
    for j in range(MKarr.shape[1]):
        if MKarr[i, j] != 0.0:
            if int(MKarr[i, j]) == 0:
                MKarr[i, j] = 5
            f.write(f"{i}\t{j}\t{int(MKarr[i, j])}\n")

f.close()


