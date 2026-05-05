#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// DRAM config
#define DRAM_NROWS 131072
#define DRAM_NCOLS 128
#define DRAM_COLSZ 8

#define DTYPE int16_t

#define min(x, y) (x) < (y) ? (x) : (y) 

typedef struct {
    DTYPE r;
    DTYPE c;
    DTYPE v;
} CooElem;



void coo_to_csr(int n_rows, int nnz, CooElem *coo,
                DTYPE *row_csr, DTYPE *col_csr, DTYPE *data_csr, DTYPE row_offset) {
    // Step 1: Initialize row_csr array
    for (int i = 0; i <= n_rows; i++) {
        row_csr[i] = 0;
    }

    // Step 2: Count number of non-zeros per row
    for (int i = 0; i < nnz; i++) {
        row_csr[coo[i].r - row_offset + 1]++;
    }

    // Step 3: Cumulative sum to get row pointers
    for (int i = 0; i < n_rows; i++) {
        // printf("row_csr = %d\n",row_csr[i]);
        row_csr[i + 1] += row_csr[i];
    }

    // Step 4: Fill col_csr and data_csr arrays
    int *temp = (int *)calloc(n_rows, sizeof(int));

    for (int i = 0; i < nnz; i++) {
        DTYPE row = coo[i].r - row_offset;
        DTYPE dest = row_csr[row] + temp[row];
        col_csr[dest] = coo[i].c;
        data_csr[dest] = coo[i].v;
        temp[row]++;
    }

    free(temp);
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


    // for (int i = 1; i < argc; i++) {
    //     if (strcmp(argv[i], "-dim") == 0) dim = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-Sp") == 0) Sp = atof(argv[++i]);
    //     else if (strcmp(argv[i], "-Tfm") == 0) Tfm = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-Tfn") == 0) Tfn = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-tau_m") == 0) tau_m = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-tau_n") == 0) tau_n = atoi(argv[++i]);
        //else if (strcmp(argv[i], "-o") == 0) strcpy(out_folder, argv[++i]);
    // }

    // I will use square tau_m and tau_n, that is why I do not need to tile matrices A and B differently


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
        fscanf(fp, "%d %d %d", &r, &c, &v);
        MKcoo[i].r = r;
        MKcoo[i].c = c; 
        MKcoo[i].v = v;
        if (r > max_row) max_row = r;
        if (c > max_col) max_col = c;
    }
    fclose(fp);

    int Mdim = max_row + 1;
    int Ndim = max_col + 1;
    int Kdim = Ndim;

    // printf("Info: M=%d\t N=%d\t K=%d\t MK_sparsity=%.1f\t KN_sparsity=%.1f\n", Mdim, Ndim, Kdim, Sp, Sp);
    printf("Tf=%d\t tau=%d\n", Tf, tau);
    printf("DRAM_Configs: BANKS: %d\t ROWS: %d\t COLS: %d\t COLSZ: %d\n", DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ);

    qsort(MKcoo, MKnnz, sizeof(CooElem), cmpCooElem);

    clock_t start_time = clock();

    // Partition the matrix A top Tfm tiles
    int curElem = 0;
    int prevElem = 0;
    for(int tm = 0; tm < Mdim; tm+=tau) {

        // the row idx boundary of this tau-tile is [tm, min(tm+tau_m,tile_cols)]
        // printf("boundary : %d\n",min(tm + tau_m, tile_cols));
        while((MKcoo[curElem].r < (min(tm + tau, Mdim))) && (curElem < MKnnz)) {
            curElem++;
        }
        int Taunnz = curElem - prevElem;
        // printf("%d\n",Taunnz);
        DTYPE* row_csr =  malloc((tau + 1) * sizeof(DTYPE));
        DTYPE* col_csr =  malloc(Taunnz * sizeof(DTYPE));
        DTYPE* data_csr = malloc(Taunnz * sizeof(DTYPE));

        coo_to_csr(tau, Taunnz, &MKcoo[prevElem], row_csr,  col_csr, data_csr, tm);
        
        prevElem = curElem;

        // DTYPE DRAM* = malloc()
        free(row_csr);
        free(col_csr);
        free(data_csr);
    }

    double elapsed_time = (clock() - start_time) * 1000 * 1000/ CLOCKS_PER_SEC;
    printf("Single matrix partitioning is done in %f us\n", elapsed_time);
    free(MKcoo);
    return 0;
}
