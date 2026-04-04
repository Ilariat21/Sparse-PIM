#!/usr/bin/env python

'''
DRAM address generator / parser
    Initially written by (97ms 10/30/2023)

'''

import math, sys, re
import configparser

protocol            = str()
bankgroups          = int()
banks_per_group     = int()
rows                = int()
columns             = int()
device_width        = int()
BL                  = int()
num_dies            = int()
bankgroup_enable    = bool()

# [system]
channel_size        = int()
channels            = int()
bus_width           = int()
address_mapping     = str()
queue_structure     = str()
row_buf_policy      = str()
cmd_queue_size      = int()
trans_queue_size    = int()
unified_queue       = bool()

# [other]
epoch_period        = int()
output_level        = int()

# [memory pos]
ch_pos              = int()
ra_pos              = int()
bg_pos              = int()
ba_pos              = int()
ro_pos              = int()
co_pos              = int()

# [memory mask]
ch_mask             = int()
ra_mask             = int()
bg_mask             = int()
ba_mask             = int()
ro_mask             = int()
co_mask             = int()

shift_bits          = int()

# parameters for main
actual_col_bits     = int()
banks               = int()


def IsGDDR():
    return(protocol == ('DDR5' or 'GDDR5X' or 'GDDR6'))

def IsHBM():
    return(protocol == ('HBM' or 'HBM2'))

def IsHMC():
    return(protocol == ('HMC'))

def IsDDR4():
    return(protocol == ('DDR4'))

def INIT(dram_config):
    global protocol
    global bankgroups
    global banks_per_group
    global rows
    global columns
    global device_width
    global BL
    global num_dies
    global bankgroup_enable

    # [system]
    global channel_size
    global channels
    global bus_width
    global address_mapping
    global queue_structure
    global row_buf_policy
    global cmd_queue_size
    global trans_queue_size
    global unified_queue

    # [other]
    global epoch_period
    global output_level

    # [memory pos]
    global ch_pos
    global ra_pos
    global bg_pos
    global ba_pos
    global ro_pos
    global co_pos

    # [memory mask]
    global ch_mask
    global ra_mask
    global bg_mask
    global ba_mask
    global ro_mask
    global co_mask

    global shift_bits

    # parameters for main
    global banks    
    global actual_col_bits

    config = configparser.ConfigParser()
    config.read(dram_config)
    # [dram_structure]
    protocol            = str(config['dram_structure']['protocol'])
    bankgroups          = int(config['dram_structure']['bankgroups'])
    banks_per_group     = int(config['dram_structure']['banks_per_group'])
    rows                = int(config['dram_structure']['rows'])
    columns             = int(config['dram_structure']['columns'])
    device_width        = int(config['dram_structure']['device_width'])
    BL                  = int(config['dram_structure']['BL'])
    num_dies            = int(config['dram_structure']['num_dies'])
    bankgroup_enable    = config.getboolean('dram_structure', 'bankgroup_enable', fallback=True)

    # [system]
    channel_size        = int(config['system']['channel_size'])
    channels            = int(config['system']['channels'])
    bus_width           = int(config['system']['bus_width'])
    address_mapping     = str(config['system']['address_mapping'])
    queue_structure     = str(config['system']['queue_structure'])
    row_buf_policy      = str(config['system']['row_buf_policy'])
    cmd_queue_size      = int(config['system']['cmd_queue_size'])
    trans_queue_size    = int(config['system']['trans_queue_size'])
    unified_queue       = bool(config['system']['unified_queue'])

    # [other]
    epoch_period        = int(config['other']['epoch_period'])
    output_level        = int(config['other']['output_level'])

    # print DRAM info
    print("\n%%%%%%%%% DRAM configuration info %%%%%%%%%\n")
    print('''[dram_structure]
    protocol          =  {0}
    bankgroups        =  {1}
    banks_per_group   =  {2}
    rows              =  {3}
    columns           =  {4}
    device_width      =  {5}
    BL                =  {6}
    num_dies          =  {7}\n'''
    .format(protocol, bankgroups, banks_per_group, rows, columns, device_width, BL, num_dies)      
    )

    print('''[system]
    channel_size      =  {}
    channels          =  {}
    bus_width         =  {}
    address_mapping   =  {}
    queue_structure   =  {}
    row_buf_policy    =  {}
    cmd_queue_size    =  {}
    trans_queue_size  =  {}
    unified_queue     =  {}\n'''
    .format(channel_size, channels, bus_width, address_mapping, queue_structure, row_buf_policy, cmd_queue_size, trans_queue_size, unified_queue)
    )

    print('''[other]
    epoch_period      =  {}
    output_level      =  {}\n'''.format(epoch_period, output_level)
    )
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")


    # set burst cycle according to protocol
    # We use burst_cycle for timing and use BL for capacity calculation
    # BL = 0 simulate perfect BW
    burst_cycle = int()
    if BL == 0:
        burst_cycle = 0
        BL = 4 if IsHBM() else 8
        print("IsHBM(): {}", IsHBM())
    else:
        burst_cycle = BL // 2

    # every protocol has a different definition of "column",
    # in DDR3/4, each column is exactly device_width bits,
    # but in GDDR5, a column is device_width * BL bits
    # and for HBM each column is device_width * 2 (prefetch)
    # as a result, different protocol has different method of calculating
    # page size, and address mapping...
    # To make life easier, we regulate the use of the term "column"
    # to only represent physical column (device width)

    if (IsGDDR()):
        columns *= BL
    elif (IsHBM()):
        columns *= 2
    
    if (not bankgroup_enable):  # aggregating all banks to one group
        banks_per_group *= bankgroups
        bankgroups = 1
    banks = int(bankgroups * banks_per_group)
    devices_per_rank = int(bus_width / device_width)
    page_size = int(columns * device_width / 8)  # page size in bytes
    megs_per_bank = int(page_size * (rows / 1024) / 1024)
    megs_per_rank = int(megs_per_bank * banks * devices_per_rank)

    if (megs_per_rank > channel_size):
        print("WARNING: Cannot create memory system of size " + str(channel_size) + \
        "MB with given device choice! Using default size " + str(megs_per_rank) + " instead!" )
        ranks = 1
        channel_size = int(megs_per_rank)
        # print('channel_size: ' + str(channel_size))
    else:
        ranks = channel_size / megs_per_rank
        channel_size = int(ranks * megs_per_rank)
        # print('channel_size: ' + str(channel_size))

    request_size_bytes = bus_width / 8 * BL
    shift_bits = int(math.log2(request_size_bytes))
    col_low_bits = int(math.log2(BL))
    actual_col_bits = int(math.log2(columns)) - col_low_bits

    field_widths = {
        "ch": int(math.log2(channels)),
        "ra": int(math.log2(ranks)),
        "bg": int(math.log2(bankgroups)),
        "ba": int(math.log2(banks_per_group)),
        "ro": int(math.log2(rows)),
        "co": actual_col_bits
    }

    fields = []
    # Split the address_mapping string into 2-character tokens and add them to the fields list
    for i in range(0, len(address_mapping), 2):
        token = address_mapping[i:i+2]
        fields.append(token)
    fields.reverse()

    field_pos = {}
    pos = 0
    while fields:
        token = fields.pop(0)
        if token not in field_widths:
            print("Unrecognized field: {}".format(token))
            raise Exception("Unrecognized field")
        field_pos[token] = int(pos)
        pos += int(field_widths[token])

    ch_pos = field_pos["ch"]
    ra_pos = field_pos["ra"]
    bg_pos = field_pos["bg"]
    ba_pos = field_pos["ba"]
    ro_pos = field_pos["ro"]
    co_pos = field_pos["co"]

    ch_mask = (1 << field_widths["ch"]) - 1
    ra_mask = (1 << field_widths["ra"]) - 1
    bg_mask = (1 << field_widths["bg"]) - 1
    ba_mask = (1 << field_widths["ba"]) - 1
    ro_mask = (1 << field_widths["ro"]) - 1
    co_mask = (1 << field_widths["co"]) - 1

    print("field mask\nch_mask: {}\tra_mask: {}\tbg_mask: {}\tba_mask: {}\tro_mask: {}\tco_mask: {}\n"\
        .format(ch_mask, ra_mask, bg_mask, ba_mask, ro_mask, co_mask))
    return banks, rows, columns//4, device_width//8

#######################################################################
### Decode
#######################################################################
# set address here
def decode(hex_addr):
    hex_addr_init=hex_addr
    hex_addr >>= shift_bits
    channel = (hex_addr >> ch_pos) & ch_mask
    rank = (hex_addr >> ra_pos) & ra_mask
    bg = (hex_addr >> bg_pos) & bg_mask
    ba = (hex_addr >> ba_pos) & ba_mask
    ro = (hex_addr >> ro_pos) & ro_mask
    co = (hex_addr >> co_pos) & co_mask
    # print("Address Decode Example")
    # print("input address: {0}\t {1}".format(hex(hex_addr_init), hex_addr_init)) 
    # print("output memory location")
    # print("ch: {0}\t rank: {1}\t bg: {2}\t bank: {3}\t ro: {4}\t co: {5}\t addr: {6} {7}"\
    #     .format(channel, rank, bg, ba, ro, co, hex(hex_addr_init),hex_addr_init))
    return(channel, rank, bg, ba, ro, co)
#######################################################################


#######################################################################
### Encode
#######################################################################
# set data location here
def encode(channel, rank, bg, ba, ro, co):
    #print(ba_mask, ro_mask, co_mask, ba_pos, ro_pos, co_pos, shift_bits)
    hex_addr=0
    hex_addr |= (channel & ch_mask) << ch_pos
    hex_addr |= (rank & ra_mask) << ra_pos
    hex_addr |= (bg & bg_mask) << bg_pos
    hex_addr |= (ba & ba_mask) << ba_pos
    hex_addr |= (ro & ro_mask) << ro_pos
    hex_addr |= (co & co_mask) << co_pos
    hex_addr <<= shift_bits
    # print("\n\n")
    # print("Address Encode Example")
    # print("input memory location")
    # print("ch: {0}\t rank: {1}\t bg: {2}\t bank: {3}\t ro: {4}\t co: {5}" .format(channel, rank, bg, ba, ro, co))
    # print('output address\t: {0}\t{1}\n'.format(hex(hex_addr),hex_addr))
    return hex(hex_addr)
#######################################################################





# def parse_DRAM_config(conffile, verbose=False):
#     global DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ    # DRAM_COLSZ: in 32b words
#     global ro_shift, ba_shift, co_shift, config

#     config = {}
#     with open(conffile, 'r') as fp:
#         for line in fp:
#             token = line.split()
#             if len(token) > 2 and token[1] == '=':
#                 try:
#                     config[token[0]] = int(token[2])
#                 except ValueError:
#                     config[token[0]] = token[2]

#     banks_per_group = config['banks_per_group']
#     bankgroups = config['bankgroups']

#     BL = config['BL']
#     if BL == 0:
#         BL = 4 if re.match('HBM', config['protocol']) else 8
#         print("IsHBM(): {}", re.match('HBM', config['protocol']))

#     bankgroup_enable = config.get('bankgroup_enable', 'false')
#     if ( re.match('[tT]rue', bankgroup_enable) ):  # aggregating all banks to one group
#         banks_per_group *= bankgroups
#         bankgroups = 1
#         config['columns'] *= BL

#     elif re.match('HBM', config['protocol']):
#         config['columns'] *= 2

#     request_size_bytes = config['bus_width'] / 8 * BL
#     shift_bits = int(math.log2(request_size_bytes))

#     DRAM_NBANKS = bankgroups * banks_per_group
#     DRAM_NROWS = config['rows']
#     DRAM_NCOLS = config['columns'] // BL
#     DRAM_COLSZ = config['bus_width'] * BL // 32     # in 32-bit words
#     print("DRAM_COLSZ: ", DRAM_COLSZ)

#     # assuming ro-ba-co ordering
#     ro_shift = int(math.log2(DRAM_NBANKS * DRAM_NCOLS * DRAM_COLSZ))
#     ba_shift = int(math.log2(DRAM_NCOLS * DRAM_COLSZ))
#     co_shift = int(math.log2(DRAM_COLSZ * shift_bits))
#     # co_shift = int(math.log2(DRAM_COLSZ * 4))
    
#     if verbose:
#         print('DRAM:\t{}\t{}\t{}\t{}\t| #ba #ro #co CoSz(word)'.format(DRAM_NBANKS, DRAM_NROWS, DRAM_NCOLS, DRAM_COLSZ))
#         print('\t{}\t{}\t{}\t| ro,ba,co_shift'.format(ro_shift, ba_shift, co_shift))


# def getaddr(ba, ro, co):
#     print("ba_shift: {}\tro_shift: {}\tco_shift: {}".format(ba_shift, ro_shift,co_shift))
#     return (ba << ba_shift) + (ro << ro_shift) + (co << co_shift)


def decimalToBinary(n): 
    return "{0:b}".format(int(n))

if __name__ == "__main__":
    #parse_DRAM_config(sys.argv[1])
    a, b, c, d = INIT(sys.argv[1])
    print("hello", a, b, c, d)
    channel, rank, bg = 0, 0, 0
    # print(hex(getaddr(0, 0, 0)))
    # print(hex(getaddr( 0, 0, 1)))
    # print(hex(getaddr( 0, 1, 0)))
    # print(hex(getaddr(0, 1, 1)))
    # print(hex(getaddr( 1, 0, 0)))
    # print(hex(getaddr( 1, 0, 1)))
    # print(hex(getaddr( 1, 1, 0)))
    # print(hex(getaddr( 1, 1, 1)))
    print(encode(channel, rank, bg, 1, 1, 0), "READ", 1)
    print(encode(channel, rank, bg, 0, 0, 0), "READ", 2)
    print(encode(channel, rank, bg, 0, 0, 30), "READ", 3)
    print(encode(channel, rank, bg, 0, 1, 0), "READ", 4)
    print(encode(channel, rank, bg, 0, 1, 1), "READ", 5)
    print(encode(channel, rank, bg, 1, 0, 0), "READ", 6)
    print(encode(channel, rank, bg, 1, 0, 1), "READ", 7)
    print(encode(channel, rank, bg, 1, 1, 0), "READ", 8)
    print(encode(channel, rank, bg, 1, 1, 1), "READ", 9)
    print(encode(channel, rank, bg, 5, 4, 3), "READ", 10)

    
    
    ba, ro, co = 5, 4, 3

    ba1 = "0"*(4-len(decimalToBinary(ba))) + decimalToBinary(ba)
    ro1 = decimalToBinary(ro)
    co1 = "0"*(5-len(decimalToBinary(co))) + decimalToBinary(co)
    addr = ro1 + ba1 + co1 + "0"*7
    num = int(addr, 2)
    hex_num = hex(num)
    print(hex_num)