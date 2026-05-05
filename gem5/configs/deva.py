import m5
from m5.objects import *
from caches import *

binary = "sparsepim"
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
system.mem_ranges = [AddrRange("8GB")]

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
system.mem_ctrl.configFile = "/home/hecate64/gem5_dramsim3/configs/dramsim.ini"
# system.mem_ctrl.dram = DRAMsim3()
#system.mem_ctrl.dram.range = system.mem_ranges[0]
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
# system.cpu.workload[0].map(0x40000000, 0x40000000, 0x40000000, cacheable=True)

print("========== Beginning simulation ==========")
exit_event = m5.simulate()

print(
    "Exiting @ tick {} because {}".format(m5.curTick(), exit_event.getCause())
)
