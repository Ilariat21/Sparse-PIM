
import subprocess, math, re

'''home/hecate64/gem5_dramsim3/time ./build/X86/gem5.opt configs/deva.py -dim 23560 -Sp 99.91'''

'''time ../gem5_deamsim3/build/X86/gem5.opt ../gem5_dramsim3/configs/deva.py -dim 2910 -Sp 97.94'''

# Input data
dims = [3363,	3562]
sparsities = [99.12,	98.74]
tf = [[1, 48]]
tau = [[32,	1024]]

# Iterate over num_rows and corresponding sparsity
flag = True
output_filename = "report_single1t.txt"

# Open the file where the results will be written
with open(output_filename, "w") as output_file:
    for i in range(len(dims)):
        for tfm, tfn in tf:
            for tau_m, tau_n in tau:
                output_file.write(f"{dims[i]}-{tau_m*tau_n*2}-{tfm*tfn}\n")
                # Prepare the command
                command = ["time", "./build/X86/gem5.opt", 
                           "configs/deva.py", 
                           '-dim', str(dims[i]), '-Sp', str(sparsities[i]), 
                           '-Tfm', str(tfm), '-Tfn', str(tfn),
                           '-tau_m', str(tau_m), '-tau_n', str(tau_n)]
                print(f"Running command: {' '.join(command)}")

                # Execute the command and capture output
                try:
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                        vr = None

                        for line in proc.stdout:
                            print(line, end='')  # Print output as it appears

                            # Look for the specific output lines and extract the numerical values
                            if "Done in" in line:
                                vr = re.search(r"Done in\s+([0-9]+\.[0-9]+)\s+us", line).group(1)
                            # elif "dram row hit rate:" in line:
                            #     mr = re.search(r"dram row hit rate:\s+([\d.]+)", line).group(1)



                        proc.wait()  # Wait for process to complete

                        # If we captured all values, write them to the file
                        if vr:
                            output_file.write(f"Latency: {vr}\n")
                            # output_file.write(f"dram row hit rate: {mr}\n")
                            output_file.write("\n")  # Blank line between entries
                            output_file.flush()

                        if proc.returncode == 0:
                            print(f"Command executed successfully")
                        else:
                            print(f"Command failed with return code {proc.returncode}")

                except subprocess.CalledProcessError as e:
                    print(f"Error executing command")
                    print("Error:", e)

                # flag = False

# Notify the user where the output is saved
print(f"Output has been saved to {output_filename}")

