import m5
from m5.objects import *
from n1_o3 import *
import math
#from m5.util import fatal

binary = "a.out"
pargs = sys.argv[1:]
system = System()
nbr_mem_ctrls = 8
system.cpu = [O3_ARM_Neoverse_N1() for i in range(np)]
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("64GB")]
def create_mem_intf(intf, r, i, intlv_bits, intlv_size, xor_low_bit):
    """
    Helper function for creating a single memoy controller from the given
    options.  This function is invoked multiple times in config_mem function
    to create an array of controllers.
    """

    import math

    intlv_low_bit = int(math.log(intlv_size, 2))

    # Use basic hashing for the channel selection, and preferably use
    # the lower tag bits from the last level cache. As we do not know
    # the details of the caches here, make an educated guess. 4 MByte
    # 4-way associative with 64 byte cache lines is 6 offset bits and
    # 14 index bits.
    if xor_low_bit:
        xor_high_bit = xor_low_bit + intlv_bits - 1
    else:
        xor_high_bit = 0

    # Create an instance so we can figure out the address
    # mapping and row-buffer size
    interface = intf()

    # Only do this for DRAMs
    if issubclass(intf, m5.objects.DRAMInterface):
        # If the channel bits are appearing after the column
        # bits, we need to add the appropriate number of bits
        # for the row buffer size
        if interface.addr_mapping.value == "RoRaBaChCo":
            # This computation only really needs to happen
            # once, but as we rely on having an instance we
            # end up having to repeat it for each and every
            # one
            rowbuffer_size = (
                interface.device_rowbuffer_size.value
                * interface.devices_per_rank.value
            )

            intlv_low_bit = int(math.log(rowbuffer_size, 2))

    # Also adjust interleaving bits for NVM attached as memory
    # Will have separate range defined with unique interleaving
    if issubclass(intf, m5.objects.NVMInterface):
        # If the channel bits are appearing after the low order
        # address bits (buffer bits), we need to add the appropriate
        # number of bits for the buffer size
        if interface.addr_mapping.value == "RoRaBaChCo":
            # This computation only really needs to happen
            # once, but as we rely on having an instance we
            # end up having to repeat it for each and every
            # one
            buffer_size = interface.per_bank_buffer_size.value

            intlv_low_bit = int(math.log(buffer_size, 2))

    # We got all we need to configure the appropriate address
    # range
    interface.range = m5.objects.AddrRange(
        r.start,
        size=r.size(),
        intlvHighBit=intlv_low_bit + intlv_bits - 1,
        xorHighBit=xor_high_bit,
        intlvBits=intlv_bits,
        intlvMatch=i,
    )
    return interface


system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "3GHz"
system.clk_domain.voltage_domain = VoltageDomain()

for cpu in system.cpu:
    cpu.clk_domain = system.clk_domain
    cpu.isa = ArmISA()
    
system.l3bus = N1_L3XBar()
system.l3bus.clk_domain  = SrcClockDomain()
system.l3bus.clk_domain.clock = str(2*np)+"GHz"
system.l3bus.clk_domain.voltage_domain = VoltageDomain()
system.l3bus.width = 32*np

system.membus = N1_SystemXBar()
system.membus.clk_domain  = SrcClockDomain()
system.membus.clk_domain.clock = str(2*nbr_mem_ctrls)+"GHz"#"2GHz"
system.membus.clk_domain.voltage_domain = VoltageDomain()
system.membus.width = 32*nbr_mem_ctrls

system.l3cache = N1_L3()
system.l3cache.cpu_side = system.l3bus.mem_side_ports
system.l3cache.mem_side = system.membus.cpu_side_ports
#system.l3cache.size = "16MB"
system.l3cache.size = "32MB"
#system.l3cache.clk_domain  = SrcClockDomain()
#system.l3cache.clk_domain.clock = str(2*np)+"GHz"
#system.l3cache.clk_domain.voltage_domain = VoltageDomain()

system.l3bus.snoop_filter.max_capacity="32MiB"
system.membus.snoop_filter.max_capacity="32MiB"

for i in range(np):
# Create L1 caches

    system.cpu[i].ndp_accel =  NDPDevA(
        ndp_ctrl=(hex(0x40000000+0x1000*i), hex(0x40001000+0x1000*i)),
        ndp_data=("0x40020000", "0x300000000"),
        max_rsze=0x40,
        max_reqs=1000,
    )

    system.cpu[i].icache = N1_ICache()
    system.cpu[i].dcache = N1_DCache()
    system.cpu[i].dcache.addr_ranges = system.mem_ranges
    system.cpu[i].cpu_id = i

    # Connect L1I cache to the CPU
    system.cpu[i].icache.cpu_side = system.cpu[i].icache_port
    
    #system.cpu[i].dcache.cpu_side = system.cpu[i].dcache_port
    system.cpu[i].ndp_accel.cpu_side = system.cpu[i].dcache_port
    system.cpu[i].dcache.cpu_side = system.cpu[i].ndp_accel.mem_side

    system.cpu[i].l2bus = L2XBar()
    #system.cpu[i].ndp_accel.dma_port = system.cpu[i].l2bus.cpu_side_ports
    system.cpu[i].ndp_accel.dma_port = system.l3bus.cpu_side_ports
    #system.cpu[i].ndp_accel.dma_port = system.membus.cpu_side_ports
    system.cpu[i].icache.mem_side = system.cpu[i].l2bus.cpu_side_ports
    system.cpu[i].dcache.mem_side = system.cpu[i].l2bus.cpu_side_ports

    # Create L2 cache
    system.cpu[i].l2cache = N1_L2()
    #system.cpu[i].l2cache.size = "512kB" ## ping-pong buffer reduces cache size to half 
        
    system.cpu[i].l2cache.cpu_side = system.cpu[i].l2bus.mem_side_ports
    system.cpu[i].l2cache.mem_side = system.l3bus.cpu_side_ports


    system.cpu[i].createInterruptController()


# Connect special port to allow read/write memory
system.system_port = system.membus.cpu_side_ports
# Create a DDR3 memory controller
intlv_bits = int(math.log(nbr_mem_ctrls, 2))
mem_ctrls = []
intlv_size = 128

for i in range(nbr_mem_ctrls):
    # Create the DRAM interface
    dram_intf = create_mem_intf(
        DDR4_3200_16x4, system.mem_ranges[0], i, intlv_bits, intlv_size, 0
    )
    # Create the controller that will drive the interface
    mem_ctrl = dram_intf.controller()
    mem_ctrls.append(mem_ctrl)

# Connect the controller to the xbar port
for i in range(len(mem_ctrls)):
    mem_ctrls[i].port = system.membus.mem_side_ports
system.mem_ctrls = mem_ctrls

system.workload = SEWorkload.init_compatible(binary)
#DDR4_2400_8x8

# Create a process for a the application
process = Process()

# Command is a list which begins with the executable (like argv)
process.cmd = [binary] + pargs.split()

# Set the cpu to use the process as its workload and create thread contexts
for i in range(np):
    system.cpu[i].workload = process
    #system.cpu.numThreads = 4
    system.cpu[i].createThreads()

# Set up the root SimObject and start the simulation
root = Root(full_system=False, system=system)

# Instantiate all of the objects we've created above
m5.instantiate()

# system.cpu[0].workload[0].map(0x40000000, 0x40000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x80000000, 0x80000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0xC0000000, 0xC0000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x100000000, 0x100000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x140000000, 0x140000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x180000000, 0x180000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x1C0000000, 0x1C0000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x200000000, 0x200000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x240000000, 0x240000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x280000000, 0x280000000, 0x40000000, cacheable=True)
# system.cpu[0].workload[0].map(0x2C0000000, 0x2C0000000, 0x40000000, cacheable=True)

print("========== Beginning simulation ==========")
exit_event = m5.simulate()

print(
    "Exiting @ tick {} because {}".format(m5.curTick(), exit_event.getCause())
)
