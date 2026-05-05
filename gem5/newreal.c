#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
// DRAM config
#define DRAM_NROWS 131072
#define DRAM_NCOLS 128
#define DRAM_COLSZ 8

#define DTYPE int16_t

#define min(x, y) (x) < (y) ? (x) : (y) 
// basically (x/8)*16+(x%8)
// fill the left half of a DRAM column
#define ADDR_CALC(x) ((x) << 1) + ((x) & 7)
typedef struct {
    DTYPE r;
    DTYPE c;
    DTYPE v;
} CooElem;

// Simulator operates with DDR4 DRAM with 128 columns of size 128 bits = 8 x FP16
// 32 bytes for a column
// DRAM_NCOLS * 32 bytes for a row 
// then move to next bank 

void coo_to_csr(int n_rows, int nnz, CooElem *coo,
                DTYPE *row_csr, DTYPE *col_csr, DTYPE *data_csr, DTYPE row_offset) {
    // Step 1: Initialize row_csr array
    for (int i = 0; i <= n_rows; i++) {
        row_csr[ADDR_CALC(i)] = 0;
        // row_csr[(i/8)*16+(i%8)] = 0;
    }
    // Step 2: Count number of non-zeros per row
    for (int i = 0; i < nnz; i++) {
        int idx = (coo[i].r - row_offset + 1);
        row_csr[ADDR_CALC(idx)]++;
        // row_csr[(idx/8)*16+(idx/8)]++;
    }
    // Step 3: Cumulative sum to get row pointers
    for (int i = 0; i < n_rows; i++) {
        // printf("row_csr = %d\n",row_csr[i]);
        row_csr[ADDR_CALC(i+1)] += row_csr[ADDR_CALC(i)];
        // row_csr[((i+1)/8)*16+((i+1)%8)] += row_csr[(i/8)*16+(i%8)];
    }
    // Step 4: Fill col_csr and data_csr arrays
    int *temp = (int *)calloc(n_rows, sizeof(int));

    for (int i = 0; i < nnz; i++) {
        DTYPE row = coo[i].r - row_offset;
        DTYPE dest = row_csr[ADDR_CALC(row)] + temp[row];
        col_csr[ADDR_CALC(dest)] = coo[i].c;
        data_csr[ADDR_CALC(dest)] = coo[i].v;
        // DTYPE dest = row_csr[(row/8)*16+(row%8)] + temp[row];
        // col_csr[(dest/8)*16+(dest%8)] = coo[i].c;
        // data_csr[(dest/8)*16+(dest%8)] = coo[i].v;
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
    int dim = 256, Tfm = 4, Tfn = 4, tau_m = 128, tau_n = 128;
    // float Sp = 90.0;
    char mat_file[256] = "nasa2910.coo.txt"; // default file name


    // for (int i = 1; i < argc; i++) {
    //     if (strcmp(argv[i], "-dim") == 0) dim = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-Sp") == 0) Sp = atof(argv[++i]);
    //     else if (strcmp(argv[i], "-Tfm") == 0) Tfm = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-Tfn") == 0) Tfn = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-tau_m") == 0) tau_m = atoi(argv[++i]);
    //     else if (strcmp(argv[i], "-tau_n") == 0) tau_n = atoi(argv[++i]);
        //else if (strcmp(argv[i], "-o") == 0) strcpy(out_folder, argv[++i]);
    // }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-mat") == 0) strcpy(mat_file, argv[++i]);
        else if (strcmp(argv[i], "-Tfm") == 0) Tfm = atoi(argv[++i]);
        else if (strcmp(argv[i], "-Tfn") == 0) Tfn = atoi(argv[++i]);
        else if (strcmp(argv[i], "-tau_m") == 0) tau_m = atoi(argv[++i]);
        else if (strcmp(argv[i], "-tau_n") == 0) tau_n = atoi(argv[++i]);
    }
    // int Mdim = dim, Ndim = dim, Kdim = dim;
    int DRAM_NBANKS = Tfm * Tfn;
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
    CooElem* KNcoo = malloc(MKnnz * sizeof(CooElem));
    int max_row = 0, max_col = 0;
    for (int i = 0; i < MKnnz; i++) {
        int r, c, v;
        fscanf(fp, "%d %d %d", &r, &c, &v);
        MKcoo[i].r = r;
        KNcoo[i].c = r;
        MKcoo[i].c = c;
        KNcoo[i].r = c;
        MKcoo[i].v = v;
        KNcoo[i].v = v;
        if (r > max_row) max_row = r;
        if (c > max_col) max_col = c;
    }
    fclose(fp);

    int Mdim = max_row + 1;
    int Ndim = max_col + 1;
    int Kdim = Ndim;
    // printf("Info: M=%d\t N=%d\t K=%d\t MK_sparsity=%.1f\t KN_sparsity=%.1f\n", Mdim, Ndim, Kdim, Sp, Sp);
    printf("Tfm=%d\t Tfn=%d\t tau_m=%d\t tau_n=%d\n", Tfm, Tfn, tau_m, tau_n);
    printf("DRAM_Configs: BANKS: %d\t ROWS: %d\t COLS: %d\t COLSZ: %d\n", DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ);

    qsort(MKcoo, MKnnz, sizeof(CooElem), cmpCooElem);
    qsort(KNcoo, MKnnz, sizeof(CooElem), cmpCooElem);

    int subM = ((Mdim + tau_m - 1) / tau_m);
    int subN = ((Mdim + tau_n - 1) / tau_n);
    int Mtiles_per_bank = (subM + Tfm - 1) / Tfm;
    int Ntiles_per_bank = (subN + Tfn - 1) / Tfn;

    // use this line one testing with real machines
    // DTYPE* BANK  = (DTYPE*)aligned_alloc( DRAM_NCOLS * DRAM_COLSZ, 131072 * DRAM_NCOLS * DRAM_COLSZ * 4);
    DTYPE* BANK = (DTYPE*)0x40000000;

    clock_t start_time = clock();

    // Partition the matrix A top Tfm tiles 
    int curElem = 0;
    int prevElem = 0;
    int MKCSRRowPTR = 0;
    int KNCSRRowPTR = 0;
    for(int tm = 0; tm < Mdim; tm+=tau_m) {
        // the row idx boundary of this tau-tile is [tm, min(tm+tau_m,tile_cols)]
        // printf("boundary : %d\n",min(tm + tau_m, tile_cols));
        while((MKcoo[curElem].r < (min(tm + tau_m, Mdim))) && (curElem < MKnnz)) {
            curElem++;
        }
        //printf("curElem %d\n", curElem);
        int Taunnz = curElem - prevElem;
        // int bank_idx = (tm / tau_m / Mtiles_per_bank);
        // printf("cur row %d\n",MKCSRRowPTR);
        // printf("processing MK tm=%d \n",tm);
        DTYPE* MKCSRrow = &BANK[MKCSRRowPTR*DRAM_NCOLS*DRAM_COLSZ*2];
        // printf("ld ld %ld %ld\n",BANK, MKCSRrow);
        MKCSRRowPTR = MKCSRRowPTR + (tau_m + 1) / 8;
        DTYPE* MKCSRcol = &BANK[MKCSRRowPTR*DRAM_NCOLS*DRAM_COLSZ*2];
        MKCSRRowPTR = MKCSRRowPTR + Taunnz / 8;
        DTYPE* MKCSRdat = &BANK[MKCSRRowPTR*DRAM_NCOLS*DRAM_COLSZ*2];
        MKCSRRowPTR = MKCSRRowPTR + Taunnz / 8;
        // printf("rowptr %d\n",MKCSRRowPTR);
        // printf("size of row_csr : %d\n", (tau_m + 1));
        // printf("size of col_csr : %d\n", Taunnz);
        // printf("cur row %d\n",MKCSRRowPTR);
        coo_to_csr(tau_m, Taunnz, &MKcoo[prevElem], MKCSRrow, MKCSRcol, MKCSRdat, tm);
        
        prevElem = curElem;

        // DTYPE DRAM* = malloc()
        // free(row_csr);
        // free(col_csr);
        // free(data_csr);
    }

    // Since matrix A and B are symmetric and therefore identical
    // we may just partition matrix A to Tfn tiles
    curElem = 0;
    prevElem = 0;
    for(int tm = 0; tm < Mdim; tm+=tau_n) {

        // the row idx boundary of this tau-tile is [tm, min(tm+tau_m,tile_cols)]
        // printf("boundary : %d\n",min(tm + tau_m, tile_cols));
        while((MKcoo[curElem].r < (min(tm + tau_n, Mdim))) && (curElem < MKnnz)) {
            curElem++;
        }
        int Taunnz = curElem - prevElem;
        // printf("%d\n",Taunnz);
        // printf("processing KN tm=%d \n",tm);
        // printf("rowptr %d\n",KNCSRRowPTR);
        DTYPE* KNCSRrow = &BANK[KNCSRRowPTR*DRAM_NCOLS*DRAM_COLSZ*2+8];
        KNCSRRowPTR = KNCSRRowPTR + (tau_n+ 1) / 8;
        // printf("rowptr %d\n",KNCSRRowPTR);
        DTYPE* KNCSRcol = &BANK[KNCSRRowPTR*DRAM_NCOLS*DRAM_COLSZ*2+8];
        KNCSRRowPTR = KNCSRRowPTR + Taunnz / 8;
        // printf("rowptr %d\n",KNCSRRowPTR);
        DTYPE* KNCSRdat = &BANK[KNCSRRowPTR*DRAM_NCOLS*DRAM_COLSZ*2+8];
        KNCSRRowPTR = KNCSRRowPTR + Taunnz / 8;
        // printf("rowptr %d\n",MKCSRRowPTR);
        // printf("size of row_csr : %d\n", (tau_m + 1));
        // printf("size of col_csr : %d\n", Taunnz);
        
        coo_to_csr(tau_n, Taunnz, &MKcoo[prevElem], KNCSRrow, KNCSRcol, KNCSRdat, tm);
        prevElem = curElem;
        
    }

    double elapsed_time = (double)(clock() - start_time) * 1000 * 1000/ CLOCKS_PER_SEC;
    printf("Pre-procesisng Done in %f us\n", elapsed_time);

    free(MKcoo);
    free(KNcoo);

    // int NUM_RES = (subM * tau_m) * (subN * tau_n);
    int NUM_TILES = subM * subN;
    int resptr = 0;
    // printf("NUM_RES %d\n",NUM_RES);

    start_time = clock();
    // memcpy(OUT, BANK, NUM_RES * sizeof(DTYPE));
    CooElem* OUT  = (DTYPE*)malloc(NUM_TILES * sizeof(DTYPE));
    for(int i = 0; i < NUM_TILES; i++) {
        if(BANK[3*i] == 0.0) {continue;}
        OUT[resptr] =  *(CooElem*)&BANK[3*i];
        // OUTR[resptr] = BANK[3*i+1];
        // OUTC[resptr] = BANK[3*i+2];
        resptr++;
    }
    elapsed_time = (double)(clock() - start_time) * 1000 * 1000/ CLOCKS_PER_SEC;

    printf("Post-procesisng Done in %f us\n", elapsed_time);
    // printf("bandwidth: %f MB/s\n", NUM_RES*2/elapsed_time);
    return 0;
}

// elapsed_time (us)
// NUM_RES * 2 bytes 
// 