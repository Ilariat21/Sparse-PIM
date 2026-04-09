import sys
import math

'''
This script is exclusively for DDR4_x16 configuration
             Initially written by (uegook 11/12/2024)
'''


# Bit shifts based on DDR4 configuration
offset_shift = 0
col_shift = 6
bank_shift = 13
rank_shift = 15
row_shift = 17  # row is from bit 17 onwards

def binToHex(bin_num):
    num = int(bin_num, 2)    # Convert binary to int
    hex_num = hex(num)       # Convert int to hexadecimal
    return hex_num

def hexToBin(hex_num):
    decimal_num = int(hex_num, 16)  # Convert hex to int
    binary_num = bin(decimal_num)   # Convert int to binary string
    binary_num = binary_num[2:]     # Remove the '0b' prefix
    return binary_num    # Pad to 32 bits if needed

def dram_encode(bank, row, column):
    rank = 0
    offset = 0
    # (channel << channel_shift) +
    addr = ((offset << offset_shift) +
            (column << col_shift) +
            (bank << bank_shift) +
            (rank << rank_shift) + 
            (row << row_shift))
    return binToHex(bin(addr)[2:])   # Remove the '0b' prefix

def dram_decode(hex_addr):
    addr = hexToBin(hex_addr)
    row = int(addr[:-row_shift], 2)
    # rank = int(addr[-row_shift:-channel_shift], 2)
    # channel = int(addr[-channel_shift:-rank_shift], 2)
    bank = int(addr[-rank_shift:-bank_shift], 2)
    column = int(addr[-bank_shift:-col_shift], 2)
    offset = int(addr[-col_shift:], 2)
    
    return bank, row, column

# # # Example usage:
# hex_address = dram_encode(bank=2, row=128, column=45)
# print("Encoded Address:", hex_address)
# hex_address = dram_encode(bank=2, row=128, column=46)
# print("Encoded Address:", hex_address)

# for i in range(5):
#     hex_address = dram_encode(channel=5, bank=2, row=i+1, column=i+2)
#     print(f"{hex_address} READ {i*2+1}")
    # print(hexToBin(hex_address))

# decoded_values = dram_decode(hex_address)
# print("Decoded Values:", decoded_values)

