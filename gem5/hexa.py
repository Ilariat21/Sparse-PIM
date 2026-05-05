for i in range(2, 66):  # 0 to 64 inclusive
    # Convert to binary string
    bin_str = bin(i)[2:]  
    
    # Append 29 zeros
    bin_with_zeros = bin_str + "0" * 29  
    
    # Convert back to integer (base 2)
    num = int(bin_with_zeros, 2)  
    
    # Convert to hexadecimal
    hex_val = hex(num)  
    
    # Print result
    # print(f"system.cpu.workload[0].map({hex_val}, {hex_val}, 0x8000000, cacheable=True)")
    print(f"{hex_val},")