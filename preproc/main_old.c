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
                DTYPE *row_csr, DTYPE *col_csr, DTYPE *data_csr) {
    // Step 1: Initialize row_csr array
    for (int i = 0; i <= n_rows; i++) {
        row_csr[i] = 0;
    }

    // Step 2: Count number of non-zeros per row
    for (int i = 0; i < nnz; i++) {
        row_csr[coo[i].r + 1]++;
    }

    // Step 3: Cumulative sum to get row pointers
    for (int i = 0; i < n_rows; i++) {
        row_csr[i + 1] += row_csr[i];
    }

    // Step 4: Fill col_csr and data_csr arrays
    int *temp = (int *)calloc(n_rows, sizeof(int));

    for (int i = 0; i < nnz; i++) {
        DTYPE row = coo[i].r;
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
    int dim = 256, Tfm = 4, Tfn = 4, tau_m = 128, tau_n = 128;
    float Sp = 90.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dim") == 0) dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-Sp") == 0) Sp = atof(argv[++i]);
        else if (strcmp(argv[i], "-Tfm") == 0) Tfm = atoi(argv[++i]);
        else if (strcmp(argv[i], "-Tfn") == 0) Tfn = atoi(argv[++i]);
        else if (strcmp(argv[i], "-tau_m") == 0) tau_m = atoi(argv[++i]);
        else if (strcmp(argv[i], "-tau_n") == 0) tau_n = atoi(argv[++i]);
        //else if (strcmp(argv[i], "-o") == 0) strcpy(out_folder, argv[++i]);
    }

    int Mdim = dim, Ndim = dim, Kdim = dim;
    int DRAM_NBANKS = Tfm * Tfn;

    printf("Info: M=%d\t N=%d\t K=%d\t MK_sparsity=%.1f\t KN_sparsity=%.1f\n", Mdim, Ndim, Kdim, Sp, Sp);
    printf("Tfm=%d\t Tfn=%d\t tau_m=%d\t tau_n=%d\n", Tfm, Tfn, tau_m, tau_n);
    printf("DRAM_Configs: BANKS: %d\t ROWS: %d\t COLS: %d\t COLSZ: %d\n", DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ);

    int MKnnz = Mdim * Kdim * (1.00f - (Sp / 100));
    //int KNnnz = Mdim * Kdim * (1.00f - (Sp / 100));

    //printf("Mknnz = %d \n", MKnnz);
    srand(42); // seed for reproducibility

    // make some dummy data in COO
    CooElem* MKcoo = malloc(MKnnz * sizeof(CooElem));
    CooElem* KNcoo = malloc(MKnnz * sizeof(CooElem));

    int MKcount = 0; 

    for (int i = 0; i < MKnnz; i++) {
        DTYPE r = rand() % Mdim;
        DTYPE c = rand() % Kdim;
        DTYPE v = rand() % 100;
        
        MKcoo[i].r = r; MKcoo[i].c = c; MKcoo[i].v = v;
        KNcoo[i].r = c; KNcoo[i].c = r; KNcoo[i].v = v;
    }

    qsort(MKcoo, MKnnz, sizeof(CooElem), cmpCooElem);
    qsort(KNcoo, MKnnz, sizeof(CooElem), cmpCooElem);
    
    int tile_rows = (Mdim + Tfm - 1) / Tfm;
    int tile_cols = (Ndim + Tfn - 1) / Tfn;

    // printf("tiled_rows = %d, tiled cols = %d\n", tile_rows, tile_cols);
    // int Tilennz = 2 * MKnnz / Tfm; // this is just a guess my not be true 
    // CooElem* (*tiledMK)[Tilennz];   
    // CooElem* (*tiledKN)[Tilennz];   
    
    // tiledMK = malloc(sizeof(CooElem[Tfm][Tilennz])); 
    // tiledKN = malloc(sizeof(CooElem[Tfn][Tilennz])); 
    clock_t start_time = clock();

    for(int m = 0; m < Tfm; m++) {
        // printf("Working on tile %d/%d\n", m+1, Tfm);
        int Tilennz = 0;   

        // count tilennz for each tile 
        for(int i = 0; i < MKnnz; i++) {
            if(MKcoo[i].r / tile_rows == m) Tilennz++;
        }

        // create temp tiledmk
        CooElem* tiledMK = malloc(Tilennz * sizeof(CooElem));

        int curElem = 0 ;
        for(int i = 0; i < MKnnz; i++) {
            if(MKcoo[i].r / tile_rows == m) {
                tiledMK[curElem].r = MKcoo[i].r % tile_rows;
                tiledMK[curElem].c = MKcoo[i].c;
                tiledMK[curElem].v = MKcoo[i].v;
                curElem++;
            }
        }  
        // printf("%d %d\n",Tilennz, curElem);
        // for(int i = 0; i < Tilennz; i++) {
        //     printf("%d %d\n", tiledMK[i].r, tiledMK[i].c);
        // }
        // proceed for each sub-tile (tau-granuality)
        curElem = 0;
        int prevElem = 0;
        for(int tm = 0; tm < tile_cols; tm+=tau_m) {

            // the row idx boundary of this tau-tile is [tm, min(tm+tau_m,tile_cols)]
            // printf("boundary : %d\n",min(tm + tau_m, tile_cols));
            while((tiledMK[curElem].r < (min(tm + tau_m, tile_cols))) && (curElem < Tilennz)) {
                curElem++;
            }
            int Taunnz = curElem - prevElem;
            // printf("%d\n",Taunnz);
            DTYPE* row_csr =  malloc((tile_rows + 1) * sizeof(DTYPE));
            DTYPE* col_csr =  malloc(Taunnz * sizeof(DTYPE));
            DTYPE* data_csr = malloc(Taunnz * sizeof(DTYPE));

            coo_to_csr(tile_rows, Taunnz, &tiledMK[prevElem], row_csr,  col_csr, data_csr);
            
            prevElem = curElem;

            // DTYPE DRAM* = malloc()
            free(row_csr);
            free(col_csr);
            free(data_csr);
        }

        free(tiledMK);
    }
    double elapsed_time = (double)(clock() - start_time) * 1000 * 1000/ CLOCKS_PER_SEC;
    printf("Done in %f us\n", elapsed_time);
    free(MKcoo);
    free(KNcoo);
    return 0;
}
