#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// In this version, matrix in only partitioned to a bank size partitions
// and each tile is compared to detect ineffectual tile-pairs (using bool: true | false) 

// DRAM config
#define DRAM_NROWS 131072
#define DRAM_NCOLS 128
#define DRAM_COLSZ 8

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
    DTYPE *col_coo;    // size = nnz (used to sdafj;aslkdf)
} CscTile;

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


void coo_to_csc(int n_cols, int nnz, CooElem *coo,
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

int cmpCooElem(const void* a, const void* b) {
    CooElem* e1 = (CooElem*)a;
    CooElem* e2 = (CooElem*)b;

    if (e1->r != e2->r)
        return e1->r - e2->r;
    else
        return e1->c - e2->c;
}

int main(int argc, char *argv[]) {
    printf("RUNNING UPDATED VERSION — STEP 1\n");

    int dim = 512, Tf = 4, tau = 256;
    // float Sp = 90.0;
    char mat_file[256] = "matrix.coo.txt"; // default file name

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

    // Count number of lines (nnz)
    int MKnnz = 0;
    char ch;
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') MKnnz++;
    }
    rewind(fp);
    
    // Allocate and read COO entries
    CooElem* MKcoo = malloc(MKnnz * sizeof(CooElem));
    int max_row = 0, max_col = 0;
    for (int i = 0; i < MKnnz; i++) {
        int r, c, v;
        int g = fscanf(fp, "%d %d %d", &r, &c, &v);
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

    DTYPE vec[sizz];   // works if Kdim is a compile-time constant (C89)

    // printf("Info: M=%d\t N=%d\t K=%d\t MK_sparsity=%.1f\t KN_sparsity=%.1f\n", Mdim, Ndim, Kdim, Sp, Sp);
    printf("Tf=%d\t tau=%d\n", Tf, tau);
    printf("DRAM_Configs: BANKS: %d\t ROWS: %d\t COLS: %d\t COLSZ: %d\n", DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ);

    // for (int i=0; i<5; i++){
    //     printf("before sort: %d %d\n", MKcoo[i].r, MKcoo[i].c);
    // }
    qsort(MKcoo, MKnnz, sizeof(CooElem), cmpCooElem);
    
    // for (int i=0; i<5; i++){
    //     printf("before sort: %d %d\n", MKcoo[i].r, MKcoo[i].c);
    // }

    clock_t start_time = clock();

    // Partition the matrix A top Tfm tiles
    int curElem = 0;
    int prevElem = 0;

    int ntiles = (Mdim + tau - 1) / tau;   // ceil division
    int Tm = (Mdim + Tf - 1) / Tf; 
    printf("# tiles: %d\n", ntiles);
    printf("Tm size: %d\n", Tm);
    
    CscTile *tiles = malloc(Tf * sizeof(CscTile));

    int tid = 0;
    for (int tm = 0; tm < Mdim; tm += Tm) {
        while ((MKcoo[curElem].r < (min(tm + Tm, Mdim))) && (curElem < MKnnz)) {
            curElem++;
        }
        int tilennz = curElem - prevElem;
        printf("nnz per tile: %d", tilennz);

        tiles[tid].ncols   = Kdim;   // or smaller at boundary
        tiles[tid].nnz     = tilennz;
        tiles[tid].col_ptr = malloc((Kdim + 1) * sizeof(MTYPE));
        tiles[tid].row_idx = malloc(tilennz * sizeof(DTYPE));
        tiles[tid].data    = malloc(tilennz * sizeof(DTYPE));
        tiles[tid].col_coo = malloc(tilennz * sizeof(DTYPE));

        coo_to_csc(Kdim, tilennz, &MKcoo[prevElem],
               tiles[tid].col_ptr,
               tiles[tid].row_idx,
               tiles[tid].data,
               tiles[tid].col_coo,
               0);

        prevElem = curElem;
        tid++;
    }

    /////////////////////////////////////////////////////////////////////


    for (int i = 0; i<5; i++){
        printf("here is random info %d %d %d\n", 
            tiles[0].col_ptr[2910], tiles[0].row_idx[i], tiles[0].data[i]);
    }
    
    int *vec1 = malloc(Tf * Tf * sizeof(bool));


    for (int i = 0; i < Tf; i++) {
        for (int j = i; j < Tf; j++) {
            // bool nonzero_found = false;
            bool nonzero_found = effectual(tiles[i].col_ptr, tiles[j].col_ptr, Kdim);
            // for (int k = 1; k < Kdim + 1; k++) {
            //     int nnz_i = tiles[i].col_ptr[k] - tiles[i].col_ptr[k - 1];
            //     int nnz_j = tiles[j].col_ptr[k] - tiles[j].col_ptr[k - 1];
            //     if (nnz_i > 0 && nnz_j > 0) {
            //         nonzero_found = true;
            //         printf("info0: %d %d %d %d %d %d\n", i, j, k, nnz_i, nnz_j, nonzero_found);
            //         break; // no need to check further
            //     }
            // }

            vec1[i * Tf + j] = nonzero_found; // store true (1) or false (0)
            vec1[j * Tf + i] = nonzero_found;
        }
    }


    for (int i=0; i<Tf*Tf; i++){
        printf("# mults : %d %d\n", i, vec1[i]);
    }
    
    
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
    
    free(tiles);
    printf("reached here\n");

    // uintptr_t bank_bases[B] = {
    // 0x40000000, 0x80000000, 0xC0000000,
    // 0x100000000, 0x140000000, 0x180000000,
    // 0x1C0000000, 0x200000000
    // };

    // DTYPE *dst = (DTYPE *)bank_bases[0];
    // DTYPE *src = tiles[tid].col_ptr;
    // size_t nbytes = (tiles[tid].ncols + 1) * sizeof(DTYPE);

    // memcpy(dst, src, nbytes);

    double elapsed_time = (clock() - start_time) * 1000 * 1000/ CLOCKS_PER_SEC;
    printf("Single matrix partitioning is done in %f us\n", elapsed_time);
    free(MKcoo);
    return 0;
}
