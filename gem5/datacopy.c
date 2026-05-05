#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define NUM_BANKS 8
#define BANK_SIZE 1024  // in bytes (just for testing)

uint8_t *bank_bases[NUM_BANKS];

void init_banks() {
    for (int i = 0; i < NUM_BANKS; i++) {
        bank_bases[i] = malloc(BANK_SIZE);
        if (!bank_bases[i]) {
            fprintf(stderr, "Failed to allocate bank %d\n", i);
            exit(1);
        }
        memset(bank_bases[i], 0, BANK_SIZE);
    }
}


void copy_to_banks(int *src, int len, int banks[], int nbanks) {
    int chunk = len / nbanks;
    for (int b = 0; b < nbanks; b++) {
        int offset = b * chunk;
        memcpy(bank_bases[banks[b]], &src[offset], chunk * sizeof(int));
    }
}

int main() {
    init_banks();

    int N = 16;
    int *A = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) A[i] = i;

    int banks_for_A[] = {0,1,2,3};
    copy_to_banks(A, N, banks_for_A, 4);

    // Check result
    for (int b = 0; b < 4; b++) {
        printf("Bank %d contents:", banks_for_A[b]);
        for (int j = 0; j < N/4; j++) {
            printf(" %d", ((int*)bank_bases[b])[j]);
        }
        printf("\n");
    }

    free(A);
    for (int i = 0; i < NUM_BANKS; i++) free(bank_bases[i]);
    return 0;
}
