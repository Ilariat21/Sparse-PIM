#!/usr/bin/env python

'''
SparsePIM Accelerator functional simulator / trace generator
GP (Gustavson Product) based compute kernel
'''

import math
import numpy as np
import os
import argparse
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from ddr_address_old import dram_encode


def compress_array_to_csr(array):  # CSR format으로 저장
    vals = []
    cols = []
    rows = [0]

    num_rows, num_cols = array.shape

    for r in range(num_rows):
        count = 0
        for c in range(num_cols):
            if array[r, c] != 0:
                vals.append(array[r, c])
                cols.append(c)
                count += 1
        rows.append(rows[-1] + count)

    return np.array(vals), np.array(cols), np.array(rows)


def read_mat(mat, flag):
    data = loadmat(mat, struct_as_record=False, squeeze_me=True)

    if 'Problem' in data:
        problem = data['Problem']
        matrix = problem.A
        name = problem.name
        if "/" in name:
            name = name.split("/")[-1]
        if isinstance(matrix, csc_matrix):
            dense_matrix = matrix.toarray()
            if flag:
                plt.spy(dense_matrix, markersize=0.1)
                plt.savefig(f"matrix_{name}.png")
                plt.clf()
            return dense_matrix
        print("Not a csc_matrix!")
    else:
        print("No 'Problem' key found")
    return None


def read_mtx(direc):
    with open(direc, 'r') as f:
        lines = f.readlines()

    first = 0
    mtx = np.zeros((100, 100), dtype=float)

    for line in lines:
        if line[0] == "%":
            continue
        token = line.split()
        if first == 0:
            first = 1
            if len(token) != 3:
                break
            m, n, _ = int(token[0]), int(token[1]), int(token[2])
            mtx = np.zeros((m, n), dtype=float)
            continue

        if len(token) < 2:
            break
        if len(token) == 2:
            r, c, v = int(token[0]), int(token[1]), 1.0
        else:
            r, c, v = int(token[0]), int(token[1]), float(token[2])
        mtx[r - 1, c - 1] = v

    return mtx


def spgemm_csr(A_row_ptr, A_col_ind, A_val,
               B_row_ptr, B_col_ind, B_val,
               ncols_B):
    nrows_A = len(A_row_ptr) - 1
    C_row_ptr = [0]
    C_col_ind = []
    C_val = []
    mac_cnt = 0

    for i in range(nrows_A):
        row_accum = {}

        for idx_A in range(A_row_ptr[i], A_row_ptr[i + 1]):
            a_col = A_col_ind[idx_A]
            a_val = A_val[idx_A]

            for idx_B in range(B_row_ptr[a_col], B_row_ptr[a_col + 1]):
                b_col = B_col_ind[idx_B]
                if b_col >= ncols_B:
                    continue
                b_val = B_val[idx_B]
                row_accum[b_col] = row_accum.get(b_col, 0.0) + (a_val * b_val)
                mac_cnt += 1

        for col, val in sorted(row_accum.items()):
            if abs(val) > 1e-12:
                C_col_ind.append(col)
                C_val.append(val)

        C_row_ptr.append(len(C_col_ind))

    return C_row_ptr, C_col_ind, C_val, mac_cnt


parser = argparse.ArgumentParser(description='SparsePIM GP-product simulator')
# parser.add_argument('-mat', '--mat', type=str, default='/home/97ms/1.research/sparse/1.gp-pim/random_100x100.mtx', help='Input matrix file')
parser.add_argument('-mat', '--mat', type=str, default='/Data4/home/97ms_local/mat/nasa2910/nasa2910.mtx', help='Input matrix file')

parser.add_argument('-Tfm', '--tiling_factor_m', type=int, default=1, help='Tiling factor of the MK matrix')
parser.add_argument('-Tfn', '--tiling_factor_n', type=int, default=16, help='Tiling factor of the KN matrix')
parser.add_argument('-tau_m', '--tau_m', type=int, default=2, help='Output buffer size (row)')
parser.add_argument('-tau_n', '--tau_n', type=int, default=1024, help='Output buffer size (col)')
parser.add_argument('-b', '--bank', type=int, default=0, help='Bank id to simulate')
parser.add_argument('-o', '--out_folder', type=str, default='trace_any', help='Output folder name')

# parser.add_argument('-ib', '--input_buffer', type=int, default=8, help='Input buffer size')
args = parser.parse_args()

out_fldr = args.out_folder
tiling_factor_m = args.tiling_factor_m
tiling_factor_n = args.tiling_factor_n
tau_m = args.tau_m
tau_n = args.tau_n
filename = args.mat
bank = args.bank
# input_buffer = args.input_buffer

if str(filename).endswith('mat'):
    MKarr = read_mat(str(filename), True)
    KNarr = read_mat(str(filename), False).T
elif str(filename).endswith('mtx'):
    MKarr = read_mtx(str(filename))
    KNarr = read_mtx(str(filename)).T
else:
    raise ValueError(f"Unsupported file type: {filename}")

Mdim, Ndim, Kdim = MKarr.shape[0], MKarr.shape[0], MKarr.shape[1]
nnz = np.count_nonzero(MKarr)
MKsp = round((1 - nnz / (Mdim * Kdim)) * 100, 2)

print(f"Info: M={Mdim}\t N={Ndim}\t K={Kdim}\t MK_sparsity={MKsp}\t NNZ={nnz}")
print(f"Tfm={tiling_factor_m}\t Tfn={tiling_factor_n}\t tau_m={tau_m}\t tau_n={tau_n}\t output_folder={out_fldr}")

itr_stn = bank // tiling_factor_n
itr_str = bank % tiling_factor_n

MKarr = MKarr[
    itr_stn * math.ceil(Mdim / tiling_factor_m):min(Mdim, (itr_stn + 1) * math.ceil(Mdim / tiling_factor_m)),
    :
]
KNarr = KNarr[
    :,
    itr_str * math.ceil(Ndim / tiling_factor_n):min(Ndim, (itr_str + 1) * math.ceil(Ndim / tiling_factor_n))
]

print("Matrix sizes:", MKarr.shape, KNarr.shape)
print('Matrix nnzs:', np.count_nonzero(MKarr), np.count_nonzero(KNarr))

cpu_res = np.dot(MKarr, KNarr)
print("CPU res shape:", cpu_res.shape)

if tau_n > KNarr.shape[1]:
    old_tau_n = tau_n
    old_tau_m = tau_m
    OB_shape = tau_m * tau_n
    tau_n = int(2 ** (math.ceil(math.log2(KNarr.shape[1]))))
    tau_m = int(OB_shape / tau_n)
    print(f"tau values are adjusted according to KN tile size: ({old_tau_m} x {old_tau_n}) => ({tau_m} x {tau_n})")

# make sub-tiles for MK array
pointers_stn, values_stn, indices_stn = [], [], []
for i in range(0, MKarr.shape[0], tau_m):
    temparr = MKarr[i:min(MKarr.shape[0], tau_m + i), :]
    vals, idxs, ptrs = compress_array_to_csr(temparr)
    values_stn.append(vals)
    indices_stn.append(idxs)
    pointers_stn.append(ptrs)

# make sub-tiles for KN array
pointers_str, values_str, indices_str = [], [], []
for i in range(0, KNarr.shape[1], tau_n):
    temparr = KNarr[:, i:min(KNarr.shape[1], tau_n + i)]
    vals, idxs, ptrs = compress_array_to_csr(temparr)
    values_str.append(vals)
    indices_str.append(idxs)
    pointers_str.append(ptrs)

if not os.path.exists(out_fldr):
    os.mkdir(out_fldr)

trace_path = f"{out_fldr}/trace_gp-{Mdim}-{MKsp}-{tau_m*tau_n*2}-{tiling_factor_m*tiling_factor_n}-{bank}.txt"
trace_list = open(trace_path, 'w')

# DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ = 1, 131072, 128, 8
DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ = 1, 65536, 128, 8
print("\nDRAM_Configs:\nBANKS: {}\t ROWS: {}\t COLS: {}\t COLSZ: {}\n".format(DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ))
DRAM_ARRAY = np.zeros((DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ), dtype=object)
bank = 0

row_num = math.ceil(KNarr.shape[1] / (DRAM_COLSZ * DRAM_NCOLS))

row_hit, row_miss = 0, 0
prev_dram_row = 9999
cycle = 1


def cycle_inc(n=1):
    global cycle
    cycle += n


def gen_mem_tra(ba, ro, co, accty):
    global row_hit, row_miss, prev_dram_row
    if ro != prev_dram_row:
        row_miss += 1
        prev_dram_row = ro
    else:
        row_hit += 1
    cycle_inc(1)
    trace_list.write(f"{dram_encode(ba, ro, co)}\t {accty}\t {cycle}\n")


def mem_get(ba, ro, co):
    return DRAM_ARRAY[ba, ro, co]


def mem_set(data, ba, ro, co, idx=None):
    if idx is not None:
        DRAM_ARRAY[ba, ro, co, idx] = data
    else:
        DRAM_ARRAY[ba, ro, co] = data

def place_ptrs(ptr1, bank, start_row):
    col, row, ptr = 0, start_row, 0
    for i in range(len(ptr1)):
        mem_set(ptr1[i], bank, row, col, ptr)
        ptr += 1
        if ptr >= DRAM_COLSZ:
            col, ptr = col + 1, 0
            if col >= DRAM_NCOLS:
                row, col = row + 1, 0
    return row if ptr == 0 and col == 0 else row + 1


def place_nzs(ptr1, ptr2, bank, start_row):
    col, row, ptr = 0, start_row, 0
    for i in range(len(ptr1)):
        mem_set(ptr1[i], bank, row, col, ptr)
        mem_set(ptr2[i], bank, row, col + 1, ptr)
        ptr += 1
        if ptr >= DRAM_COLSZ:
            col, ptr = col + 2, 0
            if col >= DRAM_NCOLS:
                row, col = row + 1, 0
    return row if ptr == 0 and col == 0 else row + 1


def addr_gen(start_idx, end_idx, start_row):
    cycle_inc()
    val_addr_list = []
    idx_addr_list = []
    start_col = start_idx // DRAM_COLSZ
    end_col = (end_idx - 1) // DRAM_COLSZ
    for i in range(start_col * 2, (end_col + 1) * 2, 2):
        val_row = start_row + i // DRAM_NCOLS
        idx_row = start_row + (i + 1) // DRAM_NCOLS
        val_col = i % DRAM_NCOLS
        idx_col = (i + 1) % DRAM_NCOLS
        val_addr_list.append([val_row, val_col])
        idx_addr_list.append([idx_row, idx_col])
    return val_addr_list, idx_addr_list


def write_buf_to_mem(buff, ba, ro, co):
    gen_mem_tra(ba, ro, co, 'WRITE')
    mem_set(buff, ba, ro, co)


def read_mem_to_buf(ba, ro, co):
    gen_mem_tra(ba, ro, co, 'READ')
    return mem_get(ba, ro, co)


def read_ptrs_from_dram(start_row, length):
    ptrs = []
    row = start_row
    col = 0
    remaining = length
    while remaining > 0:
        chunk = read_mem_to_buf(bank, row, col)
        if chunk is None:
            chunk = np.zeros(DRAM_COLSZ, dtype=float)
        take = min(remaining, DRAM_COLSZ)
        ptrs.extend(chunk[:take])
        remaining -= take
        col += 1
        if col >= DRAM_NCOLS:
            col = 0
            row += 1
    return ptrs


def read_nzs_from_dram(start_row, start_idx, end_idx):
    if end_idx <= start_idx:
        return [], []
    val_addr_list, idx_addr_list = addr_gen(start_idx, end_idx, start_row)
    val_chunks = [read_mem_to_buf(bank, r, c) for r, c in val_addr_list]
    idx_chunks = [read_mem_to_buf(bank, r, c) for r, c in idx_addr_list]
    if len(val_chunks) == 0:
        return [], []
    val_flat = np.concatenate(val_chunks).tolist()
    idx_flat = np.concatenate(idx_chunks).tolist()
    start_off = start_idx % DRAM_COLSZ
    total = end_idx - start_idx
    return val_flat[start_off:start_off + total], idx_flat[start_off:start_off + total]

# def input_buffer_cnt():
    
#     if b_vals_history and b_vals_history[-1] == b_vals:
#     same_b_vals_cnt += 1
    
#     # Check if current b_vals matches with any of previous 8 b_vals
#     # input_buffer = 128
#     for prev_b_vals in b_vals_history[-input_buffer:]:
#         if prev_b_vals == b_vals:
#             same_b_vals_8_cnt += 1
#             # print(f"{b_vals}")
#             print(f"{val_rows_str[j]}, {b_start}, {b_end}")
#             break
    
#     # Add current b_vals to history (keep only last 8)
#     b_vals_history.append(b_vals)
#     if len(b_vals_history) > input_buffer:
#         b_vals_history.pop(0)


mac_cnt = 0
sd = []

ptr_rows_stn = [0]
for i in range(len(pointers_stn)):
    ptr_rows_stn.append(place_ptrs(pointers_stn[i], bank, ptr_rows_stn[-1]))

ptr_rows_str = [ptr_rows_stn[-1]]
for i in range(len(pointers_str)):
    ptr_rows_str.append(place_ptrs(pointers_str[i], bank, ptr_rows_str[-1]))

val_rows_stn = [ptr_rows_str[-1]]
for i in range(len(values_stn)):
    val_rows_stn.append(place_nzs(values_stn[i], indices_stn[i], bank, val_rows_stn[-1]))

val_rows_str = [val_rows_stn[-1]]
for i in range(len(values_str)):
    val_rows_str.append(place_nzs(values_str[i], indices_str[i], bank, val_rows_str[-1]))


same_ptr_cnt = 0
B_row_ptr_prev = None
same_b_vals_cnt = 0
same_b_vals_8_cnt = 0  # Counter for matches with previous 8 b_vals
cache_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
# if input_buffer not in cache_sizes:
#     cache_sizes.append(input_buffer)
cache_sizes = sorted(cache_sizes)
buffer_hit_cnts = {size: 0 for size in cache_sizes}
history_limit = max(cache_sizes)
for i in tqdm(range(len(pointers_stn)), desc='GP-product tiles (M)'):
    tile_rows = min(tau_m, MKarr.shape[0] - i * tau_m)
    A_row_ptr = read_ptrs_from_dram(ptr_rows_stn[i], tile_rows + 1)
    for j in range(len(pointers_str)):
        tile_cols = min(tau_n, KNarr.shape[1] - j * tau_n)
        B_row_ptr = read_ptrs_from_dram(ptr_rows_str[j], KNarr.shape[0] + 1)
        out_buff = np.zeros((tile_rows, tile_cols), dtype=float)
        
        
        # print(B_row_ptr)
        if B_row_ptr_prev == B_row_ptr:
            same_ptr_cnt += 1
            # print(f"Same B_row_ptr as previous tile: {same_ptr_cnt} times")            
        B_row_ptr_prev = B_row_ptr

        b_vals_history = []  # Buffer to store previous b_vals
        b_vals_history_1 = []  # Buffer to store previous b_vals
        b_vals_history_2 = []  # Buffer to store previous b_vals
        b_vals_history_4 = []  # Buffer to store previous b_vals
        b_vals_history_8 = []  # Buffer to store previous b_vals
        b_vals_history_16 = []  # Buffer to store previous b_vals
        b_vals_history_32 = []  # Buffer to store previous b_vals
        b_vals_history_64 = []  # Buffer to store previous b_vals
        b_vals_history_128 = []  # Buffer to store previous b_vals


        
        for rb in range(0, tile_rows, DRAM_COLSZ):
            rb_end = min(tile_rows, rb + DRAM_COLSZ)
            vrf = np.zeros((DRAM_COLSZ, DRAM_COLSZ), dtype=float)
            current_block_idx = None
            current_cb = 0
            current_cb_end = 0

            for r in range(rb, rb_end):
                a_start = int(A_row_ptr[r])
                a_end = int(A_row_ptr[r + 1])
                if a_end <= a_start:
                    continue
                a_vals, a_cols = read_nzs_from_dram(val_rows_stn[i], a_start, a_end)

                for idx in range(len(a_cols)):
                    k = int(a_cols[idx])
                    if k < 0 or k + 1 >= len(B_row_ptr):
                        continue
                    b_start = int(B_row_ptr[k])
                    b_end = int(B_row_ptr[k + 1])
                    if b_end <= b_start:
                        continue
                    b_vals, b_cols = read_nzs_from_dram(val_rows_str[j], b_start, b_end)

                    # Check if current b_vals matches with previous 1 b_vals
                    if b_vals_history and b_vals_history[-1] == b_vals:
                        same_b_vals_cnt += 1

                    # # Check if current b_vals matches with any of previous 8 b_vals
                    # input_buffer = 1
                    # for prev_b_vals in b_vals_history[-input_buffer:]:
                    #     if prev_b_vals == b_vals:
                    #         same_b_vals_8_cnt += 1
                    #         # print(f"{b_vals}")
                    #         print(f"{val_rows_str[j]}, {b_start}, {b_end}")
                    #         break

                    # Check cache hit count for multiple input buffer sizes at once
                    for size in cache_sizes:
                        for prev_b_vals in b_vals_history[-size:]:
                            if prev_b_vals == b_vals:
                                buffer_hit_cnts[size] += 1
                                break

                    # Add current b_vals to history (keep only last 8)
                    b_vals_history.append(b_vals)
                    if len(b_vals_history) > history_limit:
                        b_vals_history.pop(0)

                    aval = a_vals[idx]
                    for t in range(len(b_cols)):
                        c = int(b_cols[t])
                        if c < 0 or c >= tile_cols:
                            continue

                        block_idx = c // DRAM_COLSZ
                        if current_block_idx is None or current_block_idx != block_idx:
                            if current_block_idx is not None:
                                for rr in range(rb, rb_end):
                                    out_buff[rr, current_cb:current_cb_end] = vrf[rr - rb, 0:current_cb_end - current_cb]

                            current_block_idx = block_idx
                            current_cb = block_idx * DRAM_COLSZ
                            current_cb_end = min(tile_cols, current_cb + DRAM_COLSZ)
                            vrf.fill(0.0)
                            for rr in range(rb, rb_end):
                                vrf[rr - rb, 0:current_cb_end - current_cb] = out_buff[rr, current_cb:current_cb_end]

                        col_off = c - current_cb
                        if col_off < 0 or col_off >= (current_cb_end - current_cb):
                            continue
                        vrf[r - rb, col_off] += aval * b_vals[t]
                        mac_cnt += 1

            if current_block_idx is not None:
                for rr in range(rb, rb_end):
                    out_buff[rr, current_cb:current_cb_end] = vrf[rr - rb, 0:current_cb_end - current_cb]

        for g in range(out_buff.shape[0]):
            for h_local in range(0, out_buff.shape[1], DRAM_COLSZ):
                atom_buff = out_buff[g, h_local:min(out_buff.shape[1], h_local + DRAM_COLSZ)]
                gg = g + tau_m * i
                h_global = h_local + tau_n * j
                dram_col = h_global % (DRAM_COLSZ * DRAM_NCOLS) // DRAM_COLSZ
                dram_row = val_rows_str[-1] + gg * row_num + h_global // (DRAM_NCOLS * DRAM_COLSZ)
                
                if not all(val == 0 for val in atom_buff):
                    tu = sum(1 for rgb in atom_buff if rgb != 0.0)
                    sd.append(DRAM_COLSZ - tu)
                    dram_atom = np.zeros(DRAM_COLSZ, dtype=float)
                    dram_atom[:len(atom_buff)] = atom_buff
                    write_buf_to_mem(dram_atom, bank, dram_row, dram_col)

ch_res = np.zeros((MKarr.shape[0], KNarr.shape[1]), dtype=float)
for i in range(ch_res.shape[0]):
    for j in range(0, ch_res.shape[1], DRAM_COLSZ):
        dram_row = val_rows_str[-1] + i * row_num + j // (DRAM_NCOLS * DRAM_COLSZ)
        dram_col = j % (DRAM_COLSZ * DRAM_NCOLS) // DRAM_COLSZ
        chunk = DRAM_ARRAY[bank, dram_row, dram_col]
        if chunk is None or isinstance(chunk, (int, float, np.number)):
            continue
        ch_res[i, j:min(ch_res.shape[1], j + DRAM_COLSZ)] = chunk[:min(ch_res.shape[1], j + DRAM_COLSZ) - j]

err_cnt = 0
print("Checking the result with CPU...")
# for i in tqdm(range(cpu_res.shape[0]), desc='Verify'):
#     for j in range(cpu_res.shape[1]):
#         cpu_v = cpu_res[i][j]
#         sim_v = ch_res[i][j]
#         if cpu_v != 0.0 or sim_v != 0.0:
#             if cpu_v == 0.0:
#                 if abs(sim_v) > 1e-4:
#                     err_cnt += 1
#             else:
#                 if abs((sim_v - cpu_v) / cpu_v) > 0.01:
#                     err_cnt += 1
for i in tqdm(range(cpu_res.shape[0])):
    for j in range(cpu_res.shape[1]):
        if cpu_res[i][j] != 0.0 or ch_res[i][j] != 0.0:
            if abs((ch_res[i][j] - cpu_res[i][j])/cpu_res[i][j])>0.01:
                err_cnt +=1
                # sys.exit(f"Error in row {i}, column {j} of the matrix: {ch_res[i][j]}!={cpu_res[i][j]}")
                print(f"Error in row {i}, column {j} of the matrix: {ch_res[i][j]}!={cpu_res[i][j]}")
print(ch_res)

print("No errors", err_cnt)
print(f"row hits: {row_hit}; row misses: {row_miss}")
if (row_hit + row_miss) > 0:
    print(f"dram row hit rate: {row_hit/(row_hit+row_miss)*100}")
print(f"# dram accesses: {row_hit + row_miss}")
print(f"# MAC ops: {mac_cnt}")
print('MK nnz:', np.count_nonzero(MKarr))
print('KN nnz:', np.count_nonzero(KNarr))

print(f"same_ptr_cnt: {same_ptr_cnt} out of {len(pointers_stn)*len(pointers_str)} tiles")
print(f"same_ptr_rate: {same_ptr_cnt/(len(pointers_stn)*len(pointers_str))*100}")
print(f"same_b_vals_cnt: {same_b_vals_cnt} out of {sum(len(a_cols) for a_cols in values_stn)*len(pointers_str)} A-B pairs")
print(f"same_b_vals_rate: {same_b_vals_cnt/(sum(len(a_cols) for a_cols in values_stn)*len(pointers_str))*100}")
# print(f"input buffer size: {input_buffer}")
total_ab_pairs = sum(len(a_cols) for a_cols in values_stn) * len(pointers_str)
print(f"buffer_hit_cnt: {same_b_vals_8_cnt}")
print(f"buffer_hit_rate: {same_b_vals_8_cnt/total_ab_pairs*100}")
print("buffer hit rates by cache size:")
for size in [1, 2, 4, 8, 16, 32, 64, 128]:
    if size in buffer_hit_cnts:
        print(f"  cache={size}: cnt={buffer_hit_cnts[size]}, rate={buffer_hit_cnts[size]/total_ab_pairs*100}")
trace_list.close()
