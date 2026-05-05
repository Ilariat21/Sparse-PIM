import m5
import sys
from m5.objects import *
from caches import *

# if len(sys.argv)>=2:
#     if sys.argv[1].split(".")[-1]=='out':
#         binary = sys.argv[1]
binary = "conv"
pargs = sys.argv[1:]

system = System()
# Simulation system
system = System()

# Clock configuration
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "3GHz"
system.clk_domain.voltage_domain = VoltageDomain()

# Memory configuration
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("16GB")]

# Create CPU
system.cpu = DerivO3CPU()

# Create L1 caches
system.cpu.icache = L1Cache(size='32kB', assoc=8)
system.cpu.dcache = L1Cache(size='64kB', assoc=8)
system.cpu.dcache.addr_ranges = system.mem_ranges

# Connect L1I cache to the CPU
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port

# Create L1 to L2 interconnect
system.l2bus = L2XBar()

# Link L1 with interconnect
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# Create L2 cache
system.l2cache = L2Cache(size='2048kB', assoc=16)

# Link L2 cache with L1 to L2 interconnect
system.l2cache.cpu_side = system.l2bus.mem_side_ports

# Create memory bus
system.membus = SystemXBar()

# Link L2 with interconnect
system.l2cache.mem_side = system.membus.cpu_side_ports

# Create interrupt controller
system.cpu.createInterruptController()

# Connect interruptions and IO with memory bus (required by X86)
if m5.defines.buildEnv["USE_X86_ISA"]:
    system.cpu.interrupts[0].pio = system.membus.mem_side_ports
    system.cpu.interrupts[0].int_master = system.membus.cpu_side_ports
    system.cpu.interrupts[0].int_slave = system.membus.mem_side_ports

# Connect special port to allow read/write memory
system.system_port = system.membus.cpu_side_ports

# Create a DDR3 memory controller
system.mem_ctrl = DRAMsim3()
system.mem_ctrl.configFile = "/home/uegook/gem5_dramsim3/configs/dramsim.ini"
# system.mem_ctrl.dram = DRAMsim3()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.workload = SEWorkload.init_compatible(binary)

# Create a process for a the application
process = Process()

# Command is a list which begins with the executable (like argv)
process.cmd = [binary] + pargs

# Set the cpu to use the process as its workload and create thread contexts
system.cpu.workload = process
system.cpu.createThreads()

# Set up the root SimObject and start the simulation
root = Root(full_system=False, system=system)

# Instantiate all of the objects we've created above
m5.instantiate()

# Dedicate upper 1GB to NDP device
# system.cpu.workload[0].map(0x10000000, 0x10000000, 0x30000000, cacheable=True)
# system.cpu.workload[0].map(0x40000000, 0x40000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0x80000000, 0x80000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0xC0000000, 0xC0000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0x100000000, 0x100000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0x140000000, 0x140000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0x180000000, 0x180000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0x1C0000000, 0x1C0000000, 0x40000000, cacheable=True)
# system.cpu.workload[0].map(0x200000000, 0x200000000, 0x40000000, cacheable=True)
system.cpu.workload[0].map(0x10000000, 0x10000000, 0x30000000, cacheable=True)
system.cpu.workload[0].map(0x40000000, 0x40000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x60000000, 0x60000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x80000000, 0x80000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0xa0000000, 0xa0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0xc0000000, 0xc0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0xe0000000, 0xe0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x100000000, 0x100000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x120000000, 0x120000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x140000000, 0x140000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x160000000, 0x160000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x180000000, 0x180000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x1a0000000, 0x1a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x1c0000000, 0x1c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x1e0000000, 0x1e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x200000000, 0x200000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x220000000, 0x220000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x240000000, 0x240000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x260000000, 0x260000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x280000000, 0x280000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x2a0000000, 0x2a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x2c0000000, 0x2c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x2e0000000, 0x2e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x300000000, 0x300000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x320000000, 0x320000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x340000000, 0x340000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x360000000, 0x360000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x380000000, 0x380000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x3a0000000, 0x3a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x3c0000000, 0x3c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x3e0000000, 0x3e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x400000000, 0x400000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x420000000, 0x420000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x440000000, 0x440000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x460000000, 0x460000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x480000000, 0x480000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x4a0000000, 0x4a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x4c0000000, 0x4c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x4e0000000, 0x4e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x500000000, 0x500000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x520000000, 0x520000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x540000000, 0x540000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x560000000, 0x560000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x580000000, 0x580000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x5a0000000, 0x5a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x5c0000000, 0x5c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x5e0000000, 0x5e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x600000000, 0x600000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x620000000, 0x620000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x640000000, 0x640000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x660000000, 0x660000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x680000000, 0x680000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x6a0000000, 0x6a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x6c0000000, 0x6c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x6e0000000, 0x6e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x700000000, 0x700000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x720000000, 0x720000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x740000000, 0x740000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x760000000, 0x760000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x780000000, 0x780000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x7a0000000, 0x7a0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x7c0000000, 0x7c0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x7e0000000, 0x7e0000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x800000000, 0x800000000, 0x8000000, cacheable=True)
system.cpu.workload[0].map(0x820000000, 0x820000000, 0x8000000, cacheable=True)

print("========== Beginning simulation ==========")
exit_event = m5.simulate()

print(
    "Exiting @ tick {} because {}".format(m5.curTick(), exit_event.getCause())
)
