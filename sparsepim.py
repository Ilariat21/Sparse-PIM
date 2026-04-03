#!/usr/bin/env python

'''
SparsePIM Accelerator functional simulator / trace generator
                      Initially written by (uegook 11/05/2024)
This version of the simulator operates with the FP16 inputs and returns FP16 outputs
The data placement assumes interleaving
Simulator operates with DDR4 DRAM with 128 columns of size 128 bits = 8 x FP16
'''


import math
import numpy as np
import sys, os, argparse, scipy.io, copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from ddr_address_old import dram_encode
import time


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

parser.add_argument('-mat', '--mat', type=str, default='/Data4/home/97ms_local/mat/nasa2910/nasa2910.mtx', help='Input matrix file')
# parser.add_argument('-mat', '--mat', type=str, default='nasa2910.mat', help='Input matrix file')
parser.add_argument('-Tfm', '--tiling_factor_m', type=int, default=1, help='Tiling factor of the MK matrix')
parser.add_argument('-Tfn', '--tiling_factor_n', type=int, default=16, help='Tiling factor of the KN matrix')
parser.add_argument('-tau_m', '--tau_m', type=int, default=2, help='Output buffer size (row)')
parser.add_argument('-tau_n', '--tau_n', type=int, default=1024, help='Output buffer size (col)')
parser.add_argument('-b', '--bank', type=int, default=0, help='Bank id to simulate')
parser.add_argument('-o', '--out_folder', type=str, default="trace_any", help="Output folder name (default: 'trace_tau')")

args = parser.parse_args()

out_fldr = args.out_folder

tiling_factor_m = args.tiling_factor_m
tiling_factor_n = args.tiling_factor_n

tau_m = args.tau_m
tau_n = args.tau_n

filename = args.mat
bank = args.bank

stn_tiles = 1
str_tiles =1

if str(filename)[-3:] == 'mat':
    MKarr = read_mat(str(filename), True)
    KNarr = read_mat(str(filename), False).T
elif str(filename)[-3:] == 'mtx':
    MKarr = read_mtx(str(filename))
    KNarr = read_mtx(str(filename)).T
# KNarr = KNarr[:, :Ndim-2]

# .transpose()
Mdim, Ndim, Kdim = MKarr.shape[0], MKarr.shape[0], MKarr.shape[1] 
nnz = np.count_nonzero(MKarr)

# KNarr = KNarr[:, :Ndim-2]
# print(KNarr.shape)
# KNarr = KNarr.T
# print(KNarr.shape)


MKsp = round((1 - nnz/(Mdim*Kdim))*100, 2)
KNsp = MKsp 

print(f"Info: M={Mdim}\t N={Ndim}\t K={Kdim}\t MK_sparsity={MKsp}\t NNZ={nnz}\n")
print(f"Tfm={tiling_factor_m}\t Tfn={tiling_factor_n}\t tau_m={tau_m}\t tau_n={tau_n}\t output_folder={out_fldr}\n")
print("Initializing the input arrays...")

# itr_stn = 0
# itr_str = bank

itr_stn = bank//tiling_factor_n
itr_str = bank%tiling_factor_n

print(itr_stn, itr_str, itr_stn*math.ceil(Mdim/tiling_factor_m), (itr_stn+1)*math.ceil(Mdim/tiling_factor_m))
print(itr_str*math.ceil(Ndim/tiling_factor_n), (itr_str+1)*math.ceil(Ndim/tiling_factor_n))

MKarr = MKarr[itr_stn*math.ceil(Mdim/tiling_factor_m):min(Mdim, (itr_stn+1)*math.ceil(Mdim/tiling_factor_m)),:]
KNarr = KNarr[:,itr_str*math.ceil(Ndim/tiling_factor_n):min(Ndim, (itr_str+1)*math.ceil(Ndim/tiling_factor_n))]


# plt.clf()

# plt.spy(MKarr, markersize=0.1)
# plt.savefig(f"matrix_A{bank}.png")
# plt.show()


# plt.clf()
# plt.spy(KNarr, markersize=0.1)
# plt.savefig(f"matrix_B{bank}.png")
# plt.show()
# plt.clf()
print("Matrix sizes:", MKarr.shape, KNarr.shape)
print('Matrix nnzs:', np.count_nonzero(MKarr), np.count_nonzero(KNarr))

cpu_res = np.dot(MKarr, KNarr)
print("CPU res shape:", cpu_res.shape)

for i in range(5):
    print(cpu_res[i, :10])

if tau_n > KNarr.shape[1]:
    print(tau_n, KNarr.shape[1], "asdf")
    old_tau = tau_n
    old_tau_m = tau_m
    OB_shape = tau_m*tau_n
    print(tau_m, tau_n, KNarr.shape)
    tau_n = int(2**(math.ceil(math.log2(KNarr.shape[1]))))
    # tau_n = int(math.ceil(KNarr.shape[1]/DRAM_COLSZ)*DRAM_COLSZ)
    tau_m = int(OB_shape/tau_n)
    print(f"tau values are adjusted according to KN tile size: ({old_tau_m} x {old_tau}) => ({tau_m} x {tau_n})\n")


pointers_stn = []
values_stn = []
indices_stn = []

# make sub-tiles for MK array
for i in range(0, MKarr.shape[0], tau_m):
    temparr = MKarr[i:min(MKarr.shape[0], tau_m+i), :]   # dense 2D array, temparr: sub-tile of size (tau_m, Kdim)
    vals, idxs, ptrs = compress_array_to_csc(temparr)
    values_stn.append(vals)
    indices_stn.append(idxs)
    pointers_stn.append(ptrs)
    temparr = []

pointers_str = []
values_str = []
indices_str = []
# make sub-tiles for KN array
for i in range(0, KNarr.shape[1], tau_n):
    temparr = KNarr[:, i:min(KNarr.shape[1], tau_n+i)]
    vals, idxs, ptrs = compress_array_to_csr(temparr)
    values_str.append(vals)
    indices_str.append(idxs)
    pointers_str.append(ptrs)

# print(pointers_stn[0][:5])
# print(pointers_str[0][500:510])

if not os.path.exists(out_fldr):
    os.mkdir(out_fldr)

trace_list = open(f"{out_fldr}/trace_real-{Mdim}-{MKsp}-{tau_m*tau_n*2}-{tiling_factor_m*tiling_factor_n}-{bank}.txt", 'w')

# DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ = 16, 32768, 32, 16
# DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ = 8, 65536, 128, 8
DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ = 1, 131072, 128, 8
print("\nDRAM_Configs:\nBANKS: {}\t ROWS: {}\t COLS: {}\t COLSZ: {}\n".format(DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ))
DRAM_ARRAY = np.zeros((DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ), dtype=object)

# how many DRAM rows are needed to store 1 output row of MN array
row_num = math.ceil(KNarr.shape[1]/(DRAM_COLSZ*DRAM_NCOLS))

def mem_get(ba, ro, co):
    return DRAM_ARRAY[ba, ro, co]

# Place elements on the column of DRAM
def mem_set(data, ba, ro, co, idx=None):      # idx in range(DRAM_COLSZ)
    if idx != None:
        DRAM_ARRAY[ba, ro, co, idx] = data    # (16b int, 16b int)
    else:
        DRAM_ARRAY[ba, ro, co] = data         # 1D-array of tuples



row_hit, row_miss = 0, 0
prev_dram_row = 9999
cycle = 1

def cycle_inc(n=1): 
    global cycle; cycle += n

def gen_mem_tra(ba, ro, co, accty):
    global row_hit, row_miss, prev_dram_row
    if ro != prev_dram_row:
        row_miss += 1
        prev_dram_row = ro
    else:
        row_hit += 1
    cycle_inc(1)
    trace_list.write('{}\t {}\t {}\n'.format(dram_encode(ba, ro, co), accty, cycle))

def read_mem_to_buf(ba, ro, co):
    gen_mem_tra(ba, ro, co, 'READ')
    return mem_get(ba, ro, co)

def write_buf_to_mem(buff, ba, ro, co, idx = None):
    gen_mem_tra(ba, ro, co, 'WRITE')
    mem_set(buff, ba, ro, co, idx)

bank = 0

# def place_ptrs(ptr1, ptr2, bank, start_row):
#     col, row, ptr = 0, start_row, 0
#     for i in range(len(ptr1)):
#         mem_set(ptr1[i], bank, row, col, ptr)
#         mem_set(ptr2[i], bank, row, col+1, ptr)
#         ptr += 1
#         if ptr >= DRAM_COLSZ:
#             col, ptr = col+2, 0
#             if col >= DRAM_NCOLS:
#                 row, col = row+1, 0
#     return row if ptr==0 and col==0 else row+1

def place_ptrs(ptr1, bank, start_row):
    col, row, ptr = 0, start_row, 0
    for i in range(len(ptr1)):
        mem_set(ptr1[i], bank, row, col, ptr)
        ptr += 1
        if ptr >= DRAM_COLSZ:
            col, ptr = col+1, 0
            if col >= DRAM_NCOLS:
                row, col = row+1, 0
    return row if ptr==0 and col==0 else row+1

def place_nzs(ptr1, ptr2, bank, start_row):
    col, row, ptr = 0, start_row, 0
    for i in range(len(ptr1)):
        mem_set(ptr1[i], bank, row, col, ptr)
        mem_set(ptr2[i], bank, row, col+1, ptr)
        ptr += 1
        if ptr >= DRAM_COLSZ:
            col, ptr = col+2, 0
            if col >= DRAM_NCOLS:
                row, col = row+1, 0
    return row if ptr==0 and col==0 else row+1

# def place_nzs(ptr1, bank, start_row):
#     col, row, ptr = 0, start_row, 0
#     for i in range(len(ptr1)):
#         mem_set(ptr1[i], bank, row, col, ptr)
#         ptr += 1
#         if ptr >= DRAM_COLSZ:
#             col, ptr = col+1, 0
#             if col >= DRAM_NCOLS:
#                 row, col = row+1, 0
#     return row if ptr==0 and col==0 else row+1

def addr_gen(start_idx, end_idx, start_row):
    cycle_inc()
    val_addr_list = []
    idx_addr_list = []
    start_idx = start_idx//DRAM_COLSZ
    end_idx = (end_idx-1)//DRAM_COLSZ
    for i in range(start_idx*2, (end_idx+1)*2, 2):
        val_row = start_row + i // DRAM_NCOLS
        idx_row = start_row + (i+1) // DRAM_NCOLS
        val_col = i % DRAM_NCOLS
        idx_col = (i+1) % DRAM_NCOLS
        val_addr_list.append([val_row, val_col])
        idx_addr_list.append([idx_row, idx_col])
    return val_addr_list, idx_addr_list

# print("row len", len(pointers_stn), len(pointers_str))

# ptr_start_rows = [0]
# for i in range(len(pointers_stn)):
#     for j in range(len(pointers_str)):
#         rrrrrr = place_ptrs(pointers_stn[i], pointers_str[j], bank, ptr_start_rows[-1])
#         # print("rfoww", rrrrrr)
#         ptr_start_rows.append(rrrrrr)



ptr_rows_stn = [0]
for i in range(len(pointers_stn)):
    ptr_rows_stn.append(place_ptrs(pointers_stn[i], bank, ptr_rows_stn[-1])) # save DRAM start row idx

ptr_rows_str = [ptr_rows_stn[-1]]
for i in range(len(pointers_str)):
    ptr_rows_str.append(place_ptrs(pointers_str[i], bank, ptr_rows_str[-1]))

val_rows_stn = [ptr_rows_str[-1]]
for i in range(len(values_stn)):
    val_rows_stn.append(place_nzs(values_stn[i], indices_stn[i], bank, val_rows_stn[-1]))

val_rows_str = [val_rows_stn[-1]]
for i in range(len(values_str)):
    val_rows_str.append(place_nzs(values_str[i], indices_str[i], bank, val_rows_str[-1]))

# print("mk",MKarr)
# print("kn",KNarr)


out_buff = np.zeros((tau_m, tau_n), dtype = float)


## 여기까지가 데이터를 DRAM에 저장하는 부분




# for i in range(len(indices_stn)):
#     print(len(indices_stn[i]))


# print("all rows:", ptr_rows_stn[-1], val_rows_stn[-1], val_rows_str[-1])

# print("this is length:", len(pointers_stn), len(pointers_str))
# print(ptr_rows_stn)
# print(ptr_rows_str)
# print(val_rows_stn)
# print(val_rows_str)

flag = 0
beg = 0

mac_cnt = 0

util = 0
eff = 0

sd = []

ptr_len = ptr_rows_stn[1] - ptr_rows_stn[0]
for i in tqdm(range(len(ptr_rows_stn)-1)):
    # print(f"Working on tile {i} / {len(ptr_rows_stn)-2}")

    for j in range(len(ptr_rows_str)-1):
        last_stn_ptr, last_str_ptr = -1, -1
        flag = 0

        for r1 in range(ptr_rows_stn[i], ptr_rows_stn[i+1]):
            if flag ==1:
                break
            for c1 in range(DRAM_NCOLS):
                if flag ==1:
                    break
                stn_ptr_chunk = read_mem_to_buf(bank, r1, c1)
                print(stn_ptr_chunk)
                r2 = r1 + (j-i)*ptr_len + ptr_rows_stn[-1]
                str_ptr_chunk = read_mem_to_buf(bank, r2, c1)
                for k in range(len(stn_ptr_chunk)):
                    if flag == 1:
                        break
                    if last_stn_ptr==-1:
                        last_str_ptr, last_stn_ptr = str_ptr_chunk[k], stn_ptr_chunk[k]
                        continue
                    stn_nnz = stn_ptr_chunk[k] - last_stn_ptr
                    str_nnz = str_ptr_chunk[k] - last_str_ptr
                    # print("a", stn_nnz, str_nnz)
                    if stn_nnz < 0 or str_nnz < 0:
                        flag = 1
                        break
                    if stn_nnz>0 and str_nnz>0:
                        stn_val_addr, stn_idx_addr = addr_gen(last_stn_ptr, stn_ptr_chunk[k], val_rows_stn[i])
                        str_val_addr, str_idx_addr = addr_gen(last_str_ptr, str_ptr_chunk[k], val_rows_str[j])
                        # print("b",stn_val_addr, str_val_addr, stn_idx_addr, str_idx_addr)
                        for p in range(0, len(stn_val_addr), stn_tiles):
                            start_stn_idx = 0
                            if p==0:
                                start_stn_idx = last_stn_ptr%DRAM_COLSZ
                            stn_val_buffer = []
                            stn_row_buffer = []
                            last_stn = stn_tiles
                            end_stn_idx = DRAM_COLSZ*stn_tiles
                            if p+stn_tiles >= len(stn_val_addr):
                                end_stn_idx = (stn_nnz+last_stn_ptr%DRAM_COLSZ)-p*DRAM_COLSZ
                                last_stn = math.ceil(end_stn_idx/DRAM_COLSZ)
                            for n in range(last_stn):
                                stn_val_buffer.append(read_mem_to_buf(bank, stn_val_addr[p+n][0], stn_val_addr[p+n][1]))
                            for n in range(last_stn):
                                stn_row_buffer.append(read_mem_to_buf(bank, stn_idx_addr[p+n][0], stn_idx_addr[p+n][1]))
                                # print("stn_val_buffer", mem_get(bank, stn_val_addr[p+n][0], stn_val_addr[p+n][1]))
                            # print("stn indices", start_stn_idx, end_stn_idx)
                            stn_val_buffer = np.concatenate(stn_val_buffer).tolist()[start_stn_idx:end_stn_idx]
                            stn_row_buffer = np.concatenate(stn_row_buffer).tolist()[start_stn_idx:end_stn_idx]
                            # print("cc", stn_row_buffer)
                            # print("stn_buffers", stn_val_buffer, stn_row_buffer)
                            for l in range(0, len(str_val_addr), str_tiles):
                                start_str_idx = 0
                                if l == 0:
                                    start_str_idx = last_str_ptr%DRAM_COLSZ
                                str_val_buffer = []
                                str_col_buffer = []
                                last_str = str_tiles
                                end_str_idx = DRAM_COLSZ*str_tiles
                                if l+str_tiles >= len(str_val_addr):
                                    end_str_idx = (str_nnz+last_str_ptr%DRAM_COLSZ)-l*DRAM_COLSZ
                                    last_str = math.ceil(end_str_idx/DRAM_COLSZ)
                                for m in range(last_str):
                                    str_val_buffer.append(read_mem_to_buf(bank, str_val_addr[l+m][0], str_val_addr[l+m][1]))
                                for m in range(last_str):
                                    str_col_buffer.append(read_mem_to_buf(bank, str_idx_addr[l+m][0], str_idx_addr[l+m][1]))
                                    # print("str_temp_buf", mem_get(bank, str_idx_addr[l+m][0], str_idx_addr[l+m][1]))
                                # vrf = np.zeros((DRAM_COLSZ*stn_tiles, DRAM_COLSZ), dtype=float)
                                vrf = np.zeros((DRAM_COLSZ, DRAM_COLSZ), dtype=float)

                                str_val_buffer = np.concatenate(str_val_buffer).tolist()[start_str_idx:end_str_idx]
                                str_col_buffer = np.concatenate(str_col_buffer).tolist()[start_str_idx:end_str_idx]
                                # print("bb",str_col_buffer)
                                for vvr in range(0, len(stn_row_buffer), vrf.shape[0]):
                                    for vvc in range(0, len(str_col_buffer), vrf.shape[1]):
                                        for vr in range(vrf.shape[0]):
                                            for vc in range(vrf.shape[1]):
                                                # cycle_inc(1)
                                                if vr+vvr < len(stn_row_buffer) and vc+vvc < len(str_col_buffer):
                                                    # print("aa", stn_row_buffer[vr+vvr], str_col_buffer[vc+vvc])
                                                    vrf[vr, vc] = out_buff[stn_row_buffer[vr+vvr], str_col_buffer[vc+vvc]]  # get data from Output buffer to VRF

                                        util += 1
                                        for r in range(min(vrf.shape[0], len(stn_row_buffer)-vvr)):
                                            for t in range(min(vrf.shape[1], len(str_col_buffer)-vvc)):
                                                temp_val = stn_val_buffer[r+vvr]*str_val_buffer[t+vvc] # mult
                                                mac_cnt += 1
                                                vrf[r, t] = vrf[r, t] + temp_val # acc
                                                eff += 1
                                        for vr in range(vrf.shape[0]):
                                            for vc in range(vrf.shape[1]):
                                                # cycle_inc(1)
                                                if vr+vvr < len(stn_row_buffer) and vc+vvc < len(str_col_buffer):
                                                    out_buff[stn_row_buffer[vr+vvr], str_col_buffer[vc+vvc]] = vrf[vr, vc] # get data from VRF to Output buffer
                                        
                    last_str_ptr, last_stn_ptr = str_ptr_chunk[k], stn_ptr_chunk[k]
            print(stn_ptr_chunk)
            time.sleep(0.1)
        for g in range(out_buff.shape[0]): # save to DRAM
            for h in range(0, out_buff.shape[1], DRAM_COLSZ):
                atom_buff = out_buff[g, h:min(out_buff.shape[1], h+DRAM_COLSZ)]
                gg = g+tau_m*i
                # print("ijgh", i, j, g, h, gg, h+tau_n*j)
                h += tau_n*j
                # if i==13 and g == 3 and h==416:
                #     print("t", gg, h, atom_buff)
                #     print(cpu_res[gg, h:h+DRAM_COLSZ])
                dram_col = h%(DRAM_COLSZ*DRAM_NCOLS)//DRAM_COLSZ
                dram_row = val_rows_str[-1] + gg*row_num + h//(DRAM_NCOLS*DRAM_COLSZ)
                if not all(val==0 for val in atom_buff):
                    tu = 0
                    for rgb in atom_buff:
                        if rgb != 0.0:
                            tu += 1
                    # print(DRAM_COLSZ-tu)
                    sd.append(DRAM_COLSZ - tu)
                    write_buf_to_mem(atom_buff, bank, dram_row, dram_col)
        out_buff = np.zeros((tau_m, tau_n), dtype=float)


# check the results
ch_res = np.zeros((MKarr.shape[0], KNarr.shape[1]), dtype=float)
for i in range(ch_res.shape[0]):
    for j in range(0, ch_res.shape[1], DRAM_COLSZ):
        dram_row = val_rows_str[-1] + i*row_num + j//(DRAM_NCOLS*DRAM_COLSZ)
        dram_col = j%(DRAM_COLSZ*DRAM_NCOLS)//DRAM_COLSZ
        ch_res[i, j:min(ch_res.shape[1], j+DRAM_COLSZ)] = DRAM_ARRAY[bank, dram_row, dram_col][:min(ch_res.shape[1], j+DRAM_COLSZ)-j]
err_cnt = 0
print("Checking the result with CPU...")
for i in tqdm(range(cpu_res.shape[0])):
    for j in range(cpu_res.shape[1]):
        if cpu_res[i][j] != 0.0 or ch_res[i][j] != 0.0:
            if abs((ch_res[i][j] - cpu_res[i][j])/cpu_res[i][j])>0.01:
                err_cnt +=1
                # sys.exit(f"Error in row {i}, column {j} of the matrix: {ch_res[i][j]}!={cpu_res[i][j]}")
                print(f"Error in row {i}, column {j} of the matrix: {ch_res[i][j]}!={cpu_res[i][j]}")

print("No errors", err_cnt)

print(f"row hits: {row_hit}; row misses: {row_miss}")
print(f"dram row hit rate: {row_hit/(row_hit+row_miss)*100}")
print(f"# dram accesses: {row_hit + row_miss}")
print(f"# MAC ops: {mac_cnt}")
print('MK nnz:', np.count_nonzero(MKarr))
print("KN nnz:", np.count_nonzero(KNarr))
print("util", util*DRAM_COLSZ*DRAM_COLSZ)
print("eff", eff)

# print("utttt", sum(sd)/(len(sd)*DRAM_COLSZ)*100)

for i in range(5):
    print(ch_res[0, :20])
# print("ratio", eff/(util*DRAM_COLSZ*DRAM_COLSZ)*100)



