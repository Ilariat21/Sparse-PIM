import numpy as np
import sys


np.random.seed(42)

if len(sys.argv) < 3:
    sys.exit('{0}: insufficient parameters\nUsage: {0}  NUM_ROWS  NUM_COLS  SPARSITY'.format(sys.argv[0]))

rows, cols, sparsity = int(sys.argv[2]), int(sys.argv[1]), float(sys.argv[3])

mat_dim = [rows, cols]

# sparsity = round((1-nnz/(rows*cols))*100, 2)
nnz = int(mat_dim[0]*mat_dim[1]*(1-sparsity/100))

print("INFO:  ROWS: {} | COLS: {} | SPARSITY: {} | NNZ: {}\n".format(mat_dim[1], mat_dim[0], 
                                                                                              sparsity, nnz))
# out_mat = open("{}/mat-CSC-{}-{}-{}.txt".format(output_fldr, 139, mat_dim[1], sparsity), 'w')
out_mat = open("mat-coo-{}-{}-{}.txt".format(mat_dim[0], mat_dim[1], sparsity), 'w')
# out_mat1 = open("{}/mat-CSR-{}-{}-{}.txt".format(output_fldr, mat_dim[0], 512, sparsity), 'w')

arr = np.zeros((rows, cols), dtype=int)

count = 0
while count < nnz:
    count += 1
    r = np.random.randint(mat_dim[0])
    c = np.random.randint(mat_dim[1])
    v = np.random.uniform(low=1, high=10)
    arr[r, c] = v

for i in range(mat_dim[0]):
    for j in range(mat_dim[1]):
        if arr[i, j] != 0:
            out_mat.write(f"{i}\t {j}\t {arr[i, j]}\n")
