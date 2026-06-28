#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// In this version, matrix is partitioned to tau size tiles,
// then each of the subtiles of one tile are compared to all subtiles of another tile to find if tile-pair is effectual or not

// DRAM config
#define DRAM_NROWS 131072
#define DRAM_NCOLS 128
#define DRAM_COLSZ 8
#define NUM_BANKS 64 // max number of banks

#define DTYPE int16_t
#define MTYPE int32_t

#define min(x, y) (x) < (y) ? (x) : (y)
#define max(x, y) (x) > (y) ? (x) : (y)

typedef struct
{
    DTYPE r;
    DTYPE c;
    DTYPE v;
} CooElem;

typedef struct
{
    int ncols;      // number of cols in this tile (tau or smaller at edge)
    int nnz;        // number of nonzeros
    MTYPE *col_ptr; // size = ncols+1
    DTYPE *row_idx; // size = nnz
    DTYPE *data;    // size = nnz
    DTYPE *col_coo; // size = nnz (used to save coo data of each subtile for further partitioning)
} CscTile;

// Comparison function by colum
int cmpByCol(const void *a, const void *b)
{
    const CooElem *x = (const CooElem *)a;
    const CooElem *y = (const CooElem *)b;
    if (x->c != y->c)
        return (x->c - y->c); // primary sort by column
    return (x->r - y->r);     // tie-breaker by row
}

void coo_to_csc(int n_cols, int nnz, CooElem *coo,
                MTYPE *col_csc, DTYPE *row_csc, 
                DTYPE *data_csc, DTYPE *col_coo, 
                DTYPE col_offset)
{
    // Sort only the portion in a column major order
    qsort(coo, nnz, sizeof(CooElem), cmpByCol);

    // Step 1: initialize col_csc
    for (int j = 0; j <= n_cols; j++)
    {
        col_csc[j] = 0;
    }

    // Step 2: count nonzeros per column
    for (int i = 0; i < nnz; i++)
    {
        col_csc[coo[i].c + 1]++;
        row_csc[i] = coo[i].r;
        data_csc[i] = coo[i].v;
    }

    // Step 3: cumulative sum
    for (int j = 0; j < n_cols; j++)
    {
        col_csc[j + 1] += col_csc[j];
    }
}

bool effectual(MTYPE *col_ptr1, MTYPE *col_ptr2, int n_cols)
{
    for (int k = 1; k < n_cols + 1; k++)
    {
        int nnz_i = col_ptr1[k] - col_ptr1[k - 1];
        int nnz_j = col_ptr2[k] - col_ptr2[k - 1];
        if (nnz_i > 0 && nnz_j > 0)
        {
            return true;
        }
    }
    return false;
}

int cmpCooElem(const void *a, const void *b)
{
    CooElem *e1 = (CooElem *)a;
    CooElem *e2 = (CooElem *)b;

    if (e1->r != e2->r)
        return e1->r - e2->r;
    else
        return e1->c - e2->c;
}

size_t write_interleaved(void *src, size_t elem_size, size_t nelems,
                         int banks[], int nbanks, uintptr_t bank_bases[], size_t offset_bytes)
{
    size_t chunk = 64; // elements per chunk (tune for cache line / row size)
    size_t total_bytes = offset_bytes;
    for (size_t i = 0; i < nelems; i += chunk)
    {
        size_t nbytes = (i + chunk <= nelems) ? chunk * elem_size : (nelems - i) * elem_size;
        for (int b = 0; b < nbanks; b++)
        {
            if (banks[b] == 99) // break if there are no banks to send data to
                break;
            uint8_t *dst = (uint8_t *)(bank_bases[banks[b]] + offset_bytes) + i * elem_size;
            memcpy(dst, (uint8_t *)src + i * elem_size, nbytes);
        }
        total_bytes += nbytes;
    }
    return total_bytes;
}

int main(int argc, char *argv[])
{
    printf("RUNNING UPDATED VERSION — STEP 1\n");

    int dim = 512, Tf = 4, tau = 128;

    char mat_file[256] = "mats/nasa2910_coo.txt"; // default file name

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-mat") == 0)
            strcpy(mat_file, argv[++i]);
        else if (strcmp(argv[i], "-Tf") == 0)
            Tf = atoi(argv[++i]);
        else if (strcmp(argv[i], "-tau") == 0)
            tau = atoi(argv[++i]);
    }

    // Tf stands for tiling factor - number of supertiles each matrix is partitioned across
    // Correspondingly number of banks is equal to Tf*Tf
    int DRAM_NBANKS = Tf * Tf;
    FILE *fp = fopen(mat_file, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to open COO file: %s\n", mat_file);
        return 1;
    }

    // Bank bases for 64 banks, beginning from
    uintptr_t bank_bases[NUM_BANKS] = {0x40000000, 0x60000000,
                                       0x80000000, 0xa0000000, 0xc0000000, 0xe0000000,
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
    while (!feof(fp))
    {
        ch = fgetc(fp);
        if (ch == '\n')
            MKnnz++;
    }
    rewind(fp);
    // Allocate and read COO entries
    CooElem *MKcoo = malloc(MKnnz * sizeof(CooElem));
    int max_row = 0, max_col = 0;
    for (int i = 0; i < MKnnz; i++)
    {
        int r, c, v;
        int g = fscanf(fp, "%d %d %d", &r, &c, &v);
        MKcoo[i].r = r;
        MKcoo[i].c = c;
        MKcoo[i].v = v;
        if (r > max_row)
            max_row = r;
        if (c > max_col)
            max_col = c;
    }
    fclose(fp);
    printf("number of lines: %d %d %d\n", MKnnz, max_row, max_col);
    int Mdim = max_row + 1;
    int Ndim = max_col + 1;
    int Kdim = Ndim;

    printf("Tf=%d\t tau=%d\n", Tf, tau);
    printf("DRAM_Configs: BANKS: %d\t ROWS: %d\t COLS: %d\t COLSZ: %d\n", DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ);

    qsort(MKcoo, MKnnz, sizeof(CooElem), cmpCooElem);

    int bank_id1 = 0;
    int bank_id2 = 0;
    int bank_itr = 0;
    bool eff_bank = false;

    // Partition the matrix A to tiles of [tau x K] size
    int curElem = 0;
    int prevElem = 0;

    int num_subtiles = (Mdim + tau - 1) / tau; // ceil division
    int Tm = (Mdim + Tf - 1) / Tf;
    int subtilesperbank = (num_subtiles + Tf - 1) / Tf;

    int *bank_to_go = malloc((2 * Tf - 1) * Tf * sizeof(int));
    for (int i = 0; i < ((2 * Tf - 1) * Tf); i++)
    {
        if (i % (2 * Tf - 1) == 0)
        {
            bank_to_go[i] = i / (2 * Tf - 1) * Tf + i / (2 * Tf - 1);
        }
        else
        {
            bank_to_go[i] = 99;
        }
    }

    int *idx = calloc(Tf, sizeof(int));
    for (int i = 0; i < Tf; i++)
    {
        idx[i] = 1;
    }

    printf("# tiles: %d\n", num_subtiles);
    printf("Tm size: %d\n", Tm);
    printf("subtiles per bank: %d\n", subtilesperbank);

    CscTile *tiles = malloc(num_subtiles * sizeof(CscTile));

    int numbanks = 0;
    int tid = 0;

    // ########## TIME starts here ##################
    // mat A of size M x K to be partitioned to sub-tiles of size [tau x K]
    // subtilesperbank = 4
    clock_t start_time = clock();
    for (int tm = 0; tm < Mdim; tm += Tm)
    {
        while ((curElem < MKnnz) && (MKcoo[curElem].r < (min(tm + Tm, Mdim))))
        {
            curElem++;
        }
        // to 16 banks, 4 tiles, 4 subtiles. 16 subtiles per matrix
        int tilennz = curElem - prevElem;

        tiles[tid].ncols = Kdim;
        tiles[tid].nnz = tilennz;
        tiles[tid].col_ptr = malloc((Kdim + 1) * sizeof(MTYPE));
        tiles[tid].row_idx = malloc(tilennz * sizeof(DTYPE));
        tiles[tid].data = malloc(tilennz * sizeof(DTYPE));
        tiles[tid].col_coo = malloc(tilennz * sizeof(DTYPE));

        coo_to_csc(Kdim, tilennz, &MKcoo[prevElem],
                   tiles[tid].col_ptr,
                   tiles[tid].row_idx,
                   tiles[tid].data,
                   tiles[tid].col_coo,
                   0);

        prevElem = curElem;
        
        //  0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 each of this is an array of size (K+1)
        //  0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
        //  M0 | M1 | M2 | M3
        //  M0 | M1 | M2 | M3
        if (tid!=0)
        {
            for (int i=0; i<tid; i++)
            {
                // This does not work yet
                bank_id1 = tid*Tf + i;
                bank_id2 = i*Tf + tid;
                bool is_effectual = effectual(tiles[i].col_ptr, tiles[tid].col_ptr, Kdim);
                if (is_effectual)
                {
                    bank_to_go[i * (2 * Tf - 1) + idx[i]] = bank_id1;
                    idx[i]++;
                    bank_to_go[i * (2 * Tf - 1) + idx[i]] = bank_id2;
                    idx[i]++;
                    // Second subtile index insertion to corresponding bank list
                    bank_to_go[tid * (2 * Tf - 1) + idx[tid]] = bank_id1;
                    idx[tid]++;
                    bank_to_go[tid * (2 * Tf - 1) + idx[tid]] = bank_id2;
                    idx[tid]++;
                }
            }
        }

        tid++;
    }

    for (int i = 0; i < ((2 * Tf - 1) * Tf); i++)
    {
        printf("tile %d bank %d\n", i / (2 * Tf - 1), bank_to_go[i]);
    }

    //     if (tid % subtilesperbank == 0 | tid == num_subtiles && tid != 0)
    //     {
    //         bank_itr = (tid + subtilesperbank - 1) / subtilesperbank - 1;
    //         // This function is only to check whether the subtile-pairs are effectual
    //         // and is there a need to transfer them to PIM-DRAM banks
    //         // This loop goes through tiles 0 to N-1, where N is the number of supertiles currently processed
    //         for (int j = 0; j < (min(num_subtiles, bank_itr * subtilesperbank)); j += subtilesperbank)
    //         {
    //             bool flag = false;
    //             // Each subtile corresponds to two banks: i*Tf + j and j*Tf + i
    //             bank_id1 = bank_itr * Tf + j / subtilesperbank;
    //             bank_id2 = j / subtilesperbank * Tf + bank_itr;
    //             // This loop goes through sub-tiles of a Nth supertile (last processed)
    //             for (int i = bank_itr * subtilesperbank; i < tid; i++)
    //             {
    //                 // If at least one subtile-pair is found to be effectual,
    //                 // all subtiles are to be transferred
    //                 if (flag)
    //                     break;
    //                 // This loop goes through each sub-tile of 0 to N-1 supertiles
    //                 for (int k = j; k < (min(num_subtiles, j + subtilesperbank)); k++)
    //                 {
    //                     // Compare each subtile-pair to find if there are any effectual multiplication in there
    //                     bool is_effectual = effectual(tiles[i].col_ptr, tiles[k].col_ptr, Kdim);
    //                     if (is_effectual)
    //                     {
    //                         // If there are multiplications to be done in this subtile-pair,
    //                         // insert subtile indices to corresponding bank list
    //                         bank_to_go[bank_itr * (2 * Tf - 1) + idx[bank_itr]] = bank_id1;
    //                         idx[bank_itr]++;
    //                         bank_to_go[bank_itr * (2 * Tf - 1) + idx[bank_itr]] = bank_id2;
    //                         idx[bank_itr]++;
    //                         // Second subtile index insertion to corresponding bank list
    //                         bank_to_go[j / subtilesperbank * (2 * Tf - 1) + idx[j / subtilesperbank]] = bank_id1;
    //                         idx[j / subtilesperbank]++;
    //                         bank_to_go[j / subtilesperbank * (2 * Tf - 1) + idx[j / subtilesperbank]] = bank_id2;
    //                         idx[j / subtilesperbank]++;
    //                         flag = true;
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    
    // double elapsed_time1 = (clock() - start_time) * 1000 * 1000 / CLOCKS_PER_SEC;
    // printf("Partitioning time: %f us\n", elapsed_time1);
    // start_time = clock();

    // // for (int i = 0; i < (Tf*Tf); i++){
    // //     printf("effectual bank %d: %d\n", i, vec[i]);
    // // }

    // for (int i = 0; i < ((2 * Tf - 1) * Tf); i++)
    // {
    //     // printf("tile %d bank %d\n", i / (2 * Tf - 1), bank_to_go[i]);

    //     for (int j = 0; j < subtilesperbank; j++)
    //     {
    //         if ((i * subtilesperbank + j) >= num_subtiles)
    //             break;
    //         size_t offset_bytes = 0;
    //         offset_bytes = write_interleaved(tiles[i * subtilesperbank + j].col_ptr, sizeof(MTYPE), (Kdim + 1),
    //                                          &bank_to_go[(i / (2 * Tf - 1))], (2 * Tf - 1), bank_bases, offset_bytes);
    //         offset_bytes = write_interleaved(tiles[i * subtilesperbank + j].row_idx, sizeof(DTYPE), tiles[i * subtilesperbank + j].nnz,
    //                                          &bank_to_go[(i / (2 * Tf - 1))], (2 * Tf - 1), bank_bases, offset_bytes);
    //         offset_bytes = write_interleaved(tiles[i * subtilesperbank + j].data, sizeof(DTYPE), tiles[i * subtilesperbank + j].nnz,
    //                                          &bank_to_go[(i / (2 * Tf - 1))], (2 * Tf - 1), bank_bases, offset_bytes);
    //     }
    // }

    double elapsed_time2 = (clock() - start_time) * 1000 * 1000 / CLOCKS_PER_SEC;
    printf("Data transfer time: %f us\n", elapsed_time2);
    // printf("Total time: %f us\n", elapsed_time1+elapsed_time2);
    free(MKcoo);
    return 0;
}
