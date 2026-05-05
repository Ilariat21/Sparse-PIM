# Read and parse the file
import sys

if len(sys.argv)<2:
    sys.exit("Usage: python {0} input_txt".format(sys.argv[0]))

filee = sys.argv[1]
data = []
with open(filee, "r") as f:
    lines = f.readlines()

# Extract config and latency
i = 0
while i < len(lines):
    line = lines[i].strip()
    if '-' in line:
        config_line = line.split()[-1]
        dim, buffer_size, banks = map(int, config_line.split('-'))

        latency_line = lines[i + 1].strip().split()[-1]
        latency = float(latency_line)

        data.append({
            "dim": dim,
            "buffer_size": buffer_size,
            "banks": banks,
            "latency": latency
        })

        i += 3  # Skip to the next block
    else:
        i += 1  # Skip non-data lines

# Sort by dim, buffer_size, banks
sorted_data = sorted(data, key=lambda x: (x["dim"], x["buffer_size"], x["banks"]))

# Write only latencies to output
with open("latencies_only.txt", "w") as f:
    for entry in sorted_data:
        f.write(f'{entry["latency"]:.6f}\n')

print("Done! Latencies are saved in 'latencies_only.txt'.")

