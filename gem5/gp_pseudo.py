def spgemm_csr(A_row_ptr, A_col_ind, A_val, 
               B_row_ptr, B_col_ind, B_val, 
               ncols_B):
    """
    Gustafson's Product (Row-wise product) using CSR format
    A: CSR format (A_row_ptr, A_col_ind, A_val)
    B: CSR format (B_row_ptr, B_col_ind, B_val)
    ncols_B: B의 전체 열 개수 (C도 같은 열 개수를 가짐)
    
    Return:
        (C_row_ptr, C_col_ind, C_val)  # CSR format
    """
    nrows_A = len(A_row_ptr) - 1
    C_row_ptr = [0]
    C_col_ind = []
    C_val = []
    
    for i in range(nrows_A):  # A의 각 row마다 수행
        row_accum = {}  # dictionary를 사용해서 임시 누적
        
        # A의 i번째 row에 대해
        for idx_A in range(A_row_ptr[i], A_row_ptr[i+1]):
            a_col = A_col_ind[idx_A]
            a_val = A_val[idx_A]
            
            # B의 a_col번째 row 가져오기
            for idx_B in range(B_row_ptr[a_col], B_row_ptr[a_col+1]):
                b_col = B_col_ind[idx_B]
                b_val = B_val[idx_B]
                
                # 누적 (C[i, b_col] += a_val * b_val)
                if b_col in row_accum:
                    row_accum[b_col] += a_val * b_val
                else:
                    row_accum[b_col] = a_val * b_val
        
        # 한 row 완료 후 CSR 형식으로 정리
        for col, val in sorted(row_accum.items()):
            if abs(val) > 1e-12:  # 0 제거
                C_col_ind.append(col)
                C_val.append(val)
        
        C_row_ptr.append(len(C_col_ind))
    
    return C_row_ptr, C_col_ind, C_val
