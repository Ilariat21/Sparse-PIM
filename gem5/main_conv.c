#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// In this version, matrix is partitioned to tau size tiles, 
// then some number of tiles are compared to others to find if they are effectual or not

// DRAM config
#define DRAM_NROWS 131072
#define DRAM_NCOLS 128
#define DRAM_COLSZ 8
#define NUM_BANKS 64

#define DTYPE int16_t
#define MTYPE int32_t

#define min(x, y) (x) < (y) ? (x) : (y) 

typedef struct {
    DTYPE r;
    DTYPE c;
    DTYPE v;
} CooElem;

typedef struct {
    int ncols;        // number of cols in this tile (tau or smaller at edge)
    int nnz;          // number of nonzeros
    MTYPE *col_ptr;   // size = ncols+1
    DTYPE *row_idx;   // size = nnz
    DTYPE *data;      // size = nnz
    DTYPE *col_coo;   // size = nnz (used to save coo data of each subtile for further partitioning)
} CscTile;

void coo_to_csc_old(int n_cols, int nnz, CooElem *coo,
                MTYPE *col_csc, DTYPE *row_csc, DTYPE *data_csc, DTYPE *col_coo, DTYPE col_offset) 
{

    // Comparison function by column
    int cmpByCol(const void *a, const void *b) {
        const CooElem *x = (const CooElem*)a;
        const CooElem *y = (const CooElem*)b;
        if (x->c != y->c) return (x->c - y->c);  // primary sort by column
        return (x->r - y->r);                     // tie-breaker by row
    }

    // for (int i=0; i<5; i++)printf("before sorting: %d %d %d\n", coo[i].r, coo[i].c, coo[i].v);
    // Sort only the portion
    qsort(coo, nnz, sizeof(CooElem), cmpByCol);

    // for (int i=0; i<5; i++)printf("after sorting: %d %d %d\n", coo[i].r, coo[i].c, coo[i].v);
    // Step 1: initialize col_csc
    for (int j = 0; j <= n_cols; j++) {
        col_csc[j] = 0;
    }

    // Step 2: count nonzeros per column
    for (int i = 0; i < nnz; i++){
        col_csc[coo[i].c+1]++;
        row_csc[i] = coo[i].r;
        data_csc[i] = coo[i].v;
    }
    // for (int i = 0; i<n_cols; i++) printf("nnz per column: %d %d\n", i, col_csc[i+1]);
    // Step 3: cumulative sum
    for (int j = 0; j < n_cols; j++) {
        // if (j > 2700 && j< 2705){
        // printf("before addition %d %d\n", j+1, col_csc[j+1]);
        // }
        
        col_csc[j + 1] += col_csc[j];
        // if (j > 2700 && j< 2705) {
        //     printf("after  addition %d %d\n", j+1, col_csc[j+1]);
        // }
    }
}

size_t write_interleaved(void *src, size_t elem_size, size_t nelems,
                       int banks[], int nbanks, uintptr_t bank_bases[], size_t offset_bytes) {
    size_t chunk = 64; // elements per chunk (tune for cache line / row size)
    size_t total_bytes = offset_bytes;
    for (size_t i = 0; i < nelems; i += chunk) {
        size_t nbytes = (i + chunk <= nelems) ? chunk * elem_size : (nelems - i) * elem_size;
        for (int b = 0; b < nbanks; b++) {
            if (banks[b]==99) break;
            uint8_t *dst = (uint8_t *)(bank_bases[banks[b]]+offset_bytes) + i * elem_size;
            memcpy(dst, (uint8_t *)src + i * elem_size, nbytes);
        }
        total_bytes += nbytes;
    }
    return total_bytes;
}


// editing this function
// void coo_to_csc(int n_cols, int nnz, CooElem *coo, 
//     int tile, int Tf, int *banks_to_go, bool flag, 
    // size_t *offset_bytes, int tilesperbank, int *bank_bases) 
void coo_to_csc(int n_cols, int nnz, CooElem *coo, 
                int tile, int Tf, int *banks_to_go, bool flag, 
                int tilesperbank, uintptr_t *bank_bases)
{
    
    MTYPE *temp_col_ptr = malloc((n_cols + 1) * sizeof(MTYPE));
    DTYPE *temp_row_idx = malloc(nnz * sizeof(DTYPE));
    DTYPE *temp_data    = malloc(nnz * sizeof(DTYPE));

    // Comparison function by column
    int cmpByCol(const void *a, const void *b) {
        const CooElem *x = (const CooElem*)a;
        const CooElem *y = (const CooElem*)b;
        if (x->c != y->c) return (x->c - y->c);  // primary sort by column
        return (x->r - y->r);                     // tie-breaker by row
    }

    // for (int i=0; i<5; i++)printf("before sorting: %d %d %d\n", coo[i].r, coo[i].c, coo[i].v);
    // Sort only the portion
    qsort(coo, nnz, sizeof(CooElem), cmpByCol);

    // for (int i=0; i<5; i++)printf("after sorting: %d %d %d\n", coo[i].r, coo[i].c, coo[i].v);
    // Step 1: initialize col_csc
    for (int j = 0; j <= n_cols; j++) {
        temp_col_ptr[j] = 0;
    }

    // Step 2: count nonzeros per column
    for (int i = 0; i < nnz; i++){
        temp_col_ptr[coo[i].c+1]++;
        temp_row_idx[i] = coo[i].r;
        temp_data[i] = coo[i].v;
    }
    // for (int i = 0; i<n_cols; i++) printf("nnz per column: %d %d\n", i, col_csc[i+1]);
    // Step 3: cumulative sum
    for (int j = 0; j < n_cols; j++) {
        // if (j > 2700 && j< 2705){
        // printf("before addition %d %d\n", j+1, col_csc[j+1]);
        // }
        
        temp_col_ptr[j + 1] += temp_col_ptr[j];
        // if (j > 2700 && j< 2705) {
        //     printf("after  addition %d %d\n", j+1, col_csc[j+1]);
        // }
    }
    // printf("tile %d\n", tile);

    int idx = 0;
    if (!flag){
        for (int i = 0; i < Tf; i++){
            int bank1 = tile + i*Tf;
            int bank2 = tile*Tf + i;
            if (bank1==bank2) {
                // printf("bank to go: %d\n", bank1);
                banks_to_go[idx] = bank1;
                idx++;
            }
            else {
                // printf("bank to go: %d\n", bank1);
                // printf("bank to go: %d\n", bank2);
                banks_to_go[idx] = bank1;
                idx++;
                banks_to_go[idx] = bank2;
                idx++;
            }   
        }
    }
    size_t offset_bytes = 0;
    for (int i = 0; i < Tf; i++){
        for (int j = 0; j < tilesperbank; j++){
            offset_bytes = write_interleaved(temp_col_ptr, sizeof(MTYPE), (n_cols + 1),
                       banks_to_go, (2*Tf-1), bank_bases, offset_bytes);
            offset_bytes = write_interleaved(temp_row_idx, sizeof(DTYPE), nnz,
                       banks_to_go, (2*Tf-1), bank_bases, offset_bytes);
            offset_bytes = write_interleaved(temp_data,    sizeof(DTYPE), nnz,
                       banks_to_go, (2*Tf-1), bank_bases, offset_bytes);      
        }

    }


    // for (int i=0; i<7; i++){
    //     printf("bank to go tile %d %d\n", tile, banks_to_go[i]);
    // }

    free(temp_col_ptr);
    free(temp_row_idx);
    free(temp_data);
}


bool effectual(MTYPE *col_ptr1, MTYPE *col_ptr2, int n_cols){
    for (int k = 1; k < n_cols + 1; k++) {
        int nnz_i = col_ptr1[k] - col_ptr1[k - 1];
        int nnz_j = col_ptr2[k] - col_ptr2[k - 1];
        if (nnz_i > 0 && nnz_j > 0) {
            return true;
        }
    }
    return false;
}

int cmpCooElem(const void* a, const void* b) {
    CooElem* e1 = (CooElem*)a;
    CooElem* e2 = (CooElem*)b;

    if (e1->r != e2->r)
        return e1->r - e2->r;
    else
        return e1->c - e2->c;
}

// void write_interleaved(void *src, size_t elem_size, size_t nelems,
//                        int banks[], int nbanks, uintptr_t bank_bases[]) {
//     size_t chunk = 64; // elements per chunk (tune for cache line / row size)
//     for (size_t i = 0; i < nelems; i += chunk) {
//         size_t nbytes = (i + chunk <= nelems) ? chunk * elem_size : (nelems - i) * elem_size;
//         for (int b = 0; b < nbanks; b++) {
//             if (banks[b]==99) break;
//             uint8_t *dst = (uint8_t *)bank_bases[banks[b]] + i * elem_size;
//             memcpy(dst, (uint8_t *)src + i * elem_size, nbytes);
//         }
//     }
// }


// #define BANK_SIZE 1024  // in bytes (just for testing)

// uint8_t *bank_bases[NUM_BANKS];

// void init_banks() {
//     for (int i = 0; i < NUM_BANKS; i++) {
//         bank_bases[i] = malloc(BANK_SIZE);
//         if (!bank_bases[i]) {
//             fprintf(stderr, "Failed to allocate bank %d\n", i);
//             exit(1);
//         }
//         memset(bank_bases[i], 0, BANK_SIZE);
//     }
// }


int main(int argc, char *argv[]) {
    printf("RUNNING UPDATED VERSION — STEP 1\n");

    int dim = 512, Tf = 4, tau = 128;
    // float Sp = 90.0;
    char mat_file[256] = "nasa2910_coo.txt"; // default file name

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-mat") == 0) strcpy(mat_file, argv[++i]);
        else if (strcmp(argv[i], "-Tf") == 0) Tf = atoi(argv[++i]);
        else if (strcmp(argv[i], "-tau") == 0) tau = atoi(argv[++i]);
    }
    // int Mdim = dim, Ndim = dim, Kdim = dim;
    int DRAM_NBANKS = Tf*Tf;
    FILE *fp = fopen(mat_file, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open COO file: %s\n", mat_file);
        return 1;
    }


    // uintptr_t bank_bases[NUM_BANKS] = {
    // 0x40000000, 0x80000000, 0xC0000000,
    // 0x100000000, 0x140000000, 0x180000000,
    // 0x1C0000000, 0x200000000
    // };
    uintptr_t bank_bases[NUM_BANKS] = {0x40000000,  0x60000000, 
        0x80000000,  0xa0000000,  0xc0000000,  0xe0000000,
        0x100000000, 0x120000000, 0x140000000, 0x160000000,
        0x180000000, 0x1a0000000, 0x1c0000000, 0x1e0000000,
        0x200000000, 0x220000000, 0x240000000, 0x260000000,
        0x280000000, 0x2a0000000, 0x2c0000000, 0x2e0000000, 
        0x300000000, 0x320000000, 0x340000000, 0x360000000,
        0x380000000, 0x3a0000000, 0x3c0000000, 0x3e0000000,
        0x400000000, 0x420000000, 0x440000000, 0x460000000,
        0x480000000, 0x4a0000000, 0x4c0000000, 0x4e0000000,
        0x500000000, 0x520000000, 0x540000000, 0x560000000,
        0x580000000, 0x5a0000000, 0x5c0000000, 0x5e0000000,
        0x600000000, 0x620000000, 0x640000000, 0x660000000,
        0x680000000, 0x6a0000000, 0x6c0000000, 0x6e0000000,
        0x700000000, 0x720000000, 0x740000000, 0x760000000,
        0x780000000, 0x7a0000000, 0x7c0000000, 0x7e0000000, 
        0x800000000, 0x820000000};

    // Count number of lines (nnz)
    int MKnnz = 0;
    char ch;
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') MKnnz++;
    }
    rewind(fp);
    // int fo = 0;
    // Allocate and read COO entries
    CooElem* MKcoo = malloc(MKnnz * sizeof(CooElem));
    int max_row = 0, max_col = 0;
    for (int i = 0; i < MKnnz; i++) {
        int r, c, v;
        int g = fscanf(fp, "%d %d %d", &r, &c, &v);
        // printf("all elements %d %d %d %d\n", fo, r, c, v);
        // fo++;
        MKcoo[i].r = r;
        MKcoo[i].c = c; 
        MKcoo[i].v = v;
        if (r > max_row) max_row = r;
        if (c > max_col) max_col = c;
    }
    fclose(fp);
    printf("number of lines: %d %d %d\n", MKnnz, max_row, max_col);
    int Mdim = max_row + 1;
    int Ndim = max_col + 1;
    int Kdim = Ndim;

    int sizz = Tf * Tf;

    int prevtile = 99;

    // bool *vec = calloc(sizz, sizeof(bool));
    // for (int t = 0; t < Tf; t++){
    //     vec[(t*Tf + t)] = true;
    // }

    // printf("Info: M=%d\t N=%d\t K=%d\t MK_sparsity=%.1f\t KN_sparsity=%.1f\n", Mdim, Ndim, Kdim, Sp, Sp);
    printf("Tf=%d\t tau=%d\n", Tf, tau);
    printf("DRAM_Configs: BANKS: %d\t ROWS: %d\t COLS: %d\t COLSZ: %d\n", DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ);

    qsort(MKcoo, MKnnz, sizeof(CooElem), cmpCooElem);

    // int bank_id1 = 0;
    // int bank_id2 = 0;
    // int bank_itr = 0;
    // bool eff_bank = false;
    bool same_tile = false;
    
    // Partition the matrix A top Tfm tiles
    int curElem = 0;
    int prevElem = 0;

    int ntiles = (Mdim + tau - 1) / tau;   // ceil division
    // int Tm = (Mdim + Tf - 1) / Tf; 
    int tilesperbank = (ntiles + Tf - 1) / Tf;

    int *bank_to_go = malloc((2*Tf-1) * Tf * sizeof(int));
    // for (int i = 0; i < ((2*Tf-1) * Tf); i++){
    //     if (i%(2*Tf-1) == 0){
    //         bank_to_go[i] = i/(2*Tf-1)*Tf + i/(2*Tf-1);
    //     }
    //     else {
    //     bank_to_go[i] = 99;
    //     }
    // }

    // int *idx = calloc(Tf, sizeof(int));
    // for (int i = 0; i < Tf; i++){
    //     idx[i] = 1;
    // }

    printf("# tiles: %d\n", ntiles);
    // printf("Tm size: %d\n", Tm);
    printf("tiles per bank: %d\n", tilesperbank);
    
    // CscTile *tiles = malloc(ntiles * sizeof(CscTile));

    // int numbanks =0;
    int tid = 0;
    size_t offset_bytes = 0;

    // ########## TIME starts here ##################
    clock_t start_time = clock();
    for (int tm = 0; tm < Mdim; tm += tau) {
        // printf("here %d %d %d %d %d %d\n", curElem, MKcoo[curElem].r, MKcoo[curElem].r, MKcoo[curElem].r, tm + tau, Mdim);
        while ((curElem < MKnnz) && (MKcoo[curElem].r < (min(tm + tau, Mdim)))) {
            // printf("bala %d %d %d\n", curElem, MKcoo[curElem].r, MKnnz);
            curElem++;
        }

        int Taunnz = curElem - prevElem;
        // printf("nnz per tile and current tile: %d %d %d %d\n", Taunnz, tid, curElem, prevElem);
    
        // tiles[tid].ncols   = Kdim;   // or smaller at boundary
        // tiles[tid].nnz     = Taunnz;
        // tiles[tid].col_ptr = malloc((Kdim + 1) * sizeof(MTYPE));
        // tiles[tid].row_idx = malloc(Taunnz * sizeof(DTYPE));
        // tiles[tid].data    = malloc(Taunnz * sizeof(DTYPE));
        // tiles[tid].col_coo = malloc(Taunnz * sizeof(DTYPE));

        int ti = tid / tilesperbank;
        if (ti == prevtile) same_tile = true;
        else {
            same_tile = false;
            prevtile = ti;
        }
        coo_to_csc(Kdim, Taunnz, &MKcoo[prevElem], ti, Tf, bank_to_go, 
            same_tile, tilesperbank, bank_bases);

        prevElem = curElem;
        tid++;
        
        // if (tid%tilesperbank == 0 | tid == ntiles && tid != 0){
        //     printf("here\n");
        //     bank_itr = (tid + tilesperbank - 1) / tilesperbank - 1;
        //     // This function is only to check whether the tile-pairs are effectual
        //     // and is there a need to transfer them to PIM-DRAM banks
        //     for (int j = 0; j < (min(ntiles, bank_itr*tilesperbank)); j+=tilesperbank){
        //         bool flag = false;
        //         // eff_bank = false;
        //         bank_id1 = bank_itr * Tf + j / tilesperbank;
        //         bank_id2 = j / tilesperbank * Tf + bank_itr;
        //         // printf("bank id %d %d %d %d\n", bank_id1, bank_id2, bank_itr, tid);
        //         for (int i = bank_itr * tilesperbank; i<tid; i++){
        //             if (flag) break;
        //             // printf("i %d %d\n", i, bank_itr);
        //             for (int k = j; k < (min(ntiles, j+tilesperbank)); k++){
        //                 bool is_effectual = effectual(tiles[i].col_ptr, tiles[k].col_ptr, Kdim);
        //                 // printf("res %d %d %d\n", i, k, is_effectual);
        //                 if (is_effectual){
        //                     // eff_bank = true;
        //                     bank_to_go[bank_itr * (2*Tf-1) + idx[bank_itr]] = bank_id1;
        //                     idx[bank_itr]++;
        //                     bank_to_go[bank_itr * (2*Tf-1) + idx[bank_itr]] = bank_id2;
        //                     idx[bank_itr]++;
        //                     bank_to_go[j/tilesperbank * (2*Tf - 1) + idx[j/tilesperbank]] = bank_id1;
        //                     idx[j/tilesperbank]++;
        //                     bank_to_go[j/tilesperbank * (2*Tf - 1) + idx[j/tilesperbank]] = bank_id2;
        //                     idx[j/tilesperbank]++;
        //                     // for (int u = 0; u =)
        //                     // vec[bank_id1] = true;
        //                     // vec[bank_id2] = true;
        //                     // printf("to check %d %d %d\n", bank_id1, bank_id2, vec[bank_id1]);
        //                     flag = true;
        //                     break;
        //                 }
        //             }
        //         }
        //         // vec[bank_id1] = eff_bank;
        //         // vec[bank_id2] = eff_bank;
        //     }
        //     // Here I start sending data to Banks

        // }
    }
    // for (int i = 0; i < (Tf*Tf); i++){
    //     printf("effectual bank %d: %d\n", i, vec[i]);
    // }

    // for (int i = 0; i < ((2*Tf-1) * Tf); i++){
    //     printf("tile %d bank %d\n", i/(2*Tf-1), bank_to_go[i]);
        
    //     for (int j = 0; j < tilesperbank; j++){
    //         if ((i*tilesperbank + j) >= ntiles) break;
    //         write_interleaved(tiles[i*tilesperbank + j].col_ptr, sizeof(MTYPE), (Kdim + 1),
    //                    &bank_to_go[(i/(2*Tf-1))], (2*Tf-1), bank_bases);
    //     }
    // }



    /////////////////////////////////////////////////////////////////////


    // for (int i = 0; i<5; i++){
    //     printf("here is random info %d %d %d\n", 
    //         tiles[0].col_ptr[2910], tiles[0].row_idx[i], tiles[0].data[i]);
    // }
    
    // int *vec1 = malloc(Tf * Tf * sizeof(bool));


    // for (int i = 0; i < Tf; i++) {
    //     for (int j = i; j < Tf; j++) {
    //         bool nonzero_found = false;
            
    //         for (int k = 1; k < Kdim + 1; k++) {
                
    //             int nnz_i = tiles[i].col_ptr[k] - tiles[i].col_ptr[k - 1];
    //             int nnz_j = tiles[j].col_ptr[k] - tiles[j].col_ptr[k - 1];
    //             if (nnz_i > 0 && nnz_j > 0) {
    //                 nonzero_found = true;
    //                 printf("info0: %d %d %d %d %d %d\n", i, j, k, nnz_i, nnz_j, nonzero_found);
    //                 break; // no need to check further
    //             }
    //         }

    //         vec1[i * Tf + j] = nonzero_found; // store true (1) or false (0)
    //         vec1[j * Tf + i] = nonzero_found;
    //     }
    // }


    // for (int i=0; i<Tf*Tf; i++){
    //     printf("# mults : %d %d\n", i, vec1[i]);
    // }
    
    
    // int tid = 0;
    // for (int tm = 0; tm < Mdim; tm += tau) {
    //     while ((MKcoo[curElem].r < (min(tm + tau, Mdim))) && (curElem < MKnnz)) {
    //         curElem++;
    //     }
    //     int Taunnz = curElem - prevElem;

    //     tiles[tid].ncols   = Kdim;   // or smaller at boundary
    //     tiles[tid].nnz     = Taunnz;
    //     tiles[tid].col_ptr = malloc((Kdim + 1) * sizeof(DTYPE));
    //     tiles[tid].row_idx = malloc(Taunnz * sizeof(DTYPE));
    //     tiles[tid].data    = malloc(Taunnz * sizeof(DTYPE));

    //     coo_to_csc(Kdim, Taunnz, &MKcoo[prevElem],
    //            tiles[tid].col_ptr,
    //            tiles[tid].row_idx,
    //            tiles[tid].data,
    //            0);

    //     prevElem = curElem;
    //     tid++;
    // }
    
    // printf("reached here\n");

    

    // DTYPE *dst = (DTYPE *)bank_bases[0];
    // DTYPE *src = tiles[tid].col_ptr;
    // size_t nbytes = (tiles[tid].ncols + 1) * sizeof(DTYPE);

    // memcpy(dst, src, nbytes);

    double elapsed_time = (clock() - start_time) * 1000 * 1000 / CLOCKS_PER_SEC;
    printf("Pre-processing time: %f us\n", elapsed_time);
    free(MKcoo);
    return 0;
}



                // uint8_t *dst = (uint8_t *)bank_bases[0];
                // printf("destination %d\n", *dst);

                // // copy col_ptr (int32_t)
                // size_t nbytes_col = (tiles[tid].ncols + 1) * sizeof(MTYPE);
                // memcpy(dst, tiles[tid].col_ptr, nbytes_col);
                // dst += nbytes_col;

                // // copy row_idx (int16_t)
                // size_t nbytes_row = tiles[tid].nnz * sizeof(DTYPE);
                // memcpy(dst, tiles[tid].row_idx, nbytes_row);
                // dst += nbytes_row;

                // // copy data (int16_t)
                // size_t nbytes_data = tiles[tid].nnz * sizeof(DTYPE);
                // memcpy(dst, tiles[tid].data, nbytes_data);
                // dst += nbytes_data;