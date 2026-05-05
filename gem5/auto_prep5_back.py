
import subprocess, math, re, os
import threading
from concurrent.futures import ThreadPoolExecutor
import time

'''home/hecate64/gem5_dramsim3/time ./build/X86/gem5.opt configs/deva.py -dim 23560 -Sp 99.91'''

'''time ../gem5_deamsim3/build/X86/gem5.opt ../gem5_dramsim3/configs/deva.py -dim 2910 -Sp 97.94'''

# Input data
dims = [13514,	23560]
sparsities = [99.81,	99.91]
# tf = [[1, 48]]
# tau = [[32,	1024]]

tfs = [2,4,8,16,32]
taus = [32,64,128,256,512,1024]
mats = ["nasa2910","raefsky1","ex9","bcsstk24","cavity26","crystk01","s3rmt3m3","t2dah_a","poisson3Da","af23560"]


# tfs = [2]
# taus = [32]
# mats = ["nasa2910"]



# MATRIX_PATH = "/Data4/home/97ms_local/mat"
MATRIX_PATH = "mat"

# Create report directory if it doesn't exist
os.makedirs("report", exist_ok=True)

# Create dictionaries to store results for table format
pt_results = {}  # Partitioning time results
dt_results = {}  # Data transfer time results

# Process each matrix separately
for mat in mats:
    print(f"Processing matrix: {mat}")
    
    # Initialize results dictionaries for this matrix
    for tf in tfs:
        pt_results[tf] = {}
        dt_results[tf] = {}
        for tau in taus:
            pt_results[tf][tau] = None
            dt_results[tf][tau] = None
    
    # Run experiments for all combinations
    for tf in tfs:
        for tau in taus:
            print(f"Running: {mat}, tf={tf}, tau={tau}")
            
            # Prepare the command
            command = ["time", "./build/X86-self/gem5.opt", 
                       "configs/deva2.py",
                       "-mat", str(MATRIX_PATH+"/"+mat+".mtx"),
                       '-tau', str(tau), '-Tf', str(tf)
                       ]
            print(f"Running command: {' '.join(command)}")

            # Execute the command and capture output
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                    pt = None
                    dt = None

                    for line in proc.stdout:
                        print(line, end='')  # Print output as it appears

                        # Look for the specific output lines and extract the numerical values
                        if "Partitioning time" in line:
                            pt = re.search(r"Partitioning time:\s+([0-9]+\.[0-9]+)\s+us", line).group(1)
                        elif "Data transfer time" in line:
                            dt = re.search(r"Data transfer time:\s+([0-9]+\.[0-9]+)\s+us", line).group(1)

                    proc.wait()  # Wait for process to complete

                    # Store the results in the dictionaries
                    if pt:
                        pt_results[tf][tau] = float(pt)
                    else:
                        pt_results[tf][tau] = "N/A"
                        
                    if dt:
                        dt_results[tf][tau] = float(dt)
                    else:
                        dt_results[tf][tau] = "N/A"

                    if proc.returncode == 0:
                        print(f"Command executed successfully")
                    else:
                        print(f"Command failed with return code {proc.returncode}")

            except subprocess.CalledProcessError as e:
                print(f"Error executing command")
                print("Error:", e)
                pt_results[tf][tau] = "ERROR"
                dt_results[tf][tau] = "ERROR"

    # Write combined file with both partitioning time and data transfer time
    combined_filename = f"report/{mat}_results.txt"
    
    with open(combined_filename, "w") as output_file:
        output_file.write(f"Matrix: {mat}\n")
        output_file.write("="*60 + "\n\n")
        
        # Write Partitioning Time table
        output_file.write("Partitioning Time (us)\n")
        output_file.write("-" * 30 + "\n")
        
        # Header row (tau values)
        output_file.write("tf\\tau\t")
        for tau in taus:
            output_file.write(f"{tau}\t")
        output_file.write("\n")
        
        # Data rows for partitioning time
        for tf in tfs:
            output_file.write(f"{tf}\t")
            for tau in taus:
                if pt_results[tf][tau] is not None:
                    if isinstance(pt_results[tf][tau], float):
                        output_file.write(f"{pt_results[tf][tau]:.2f}\t")
                    else:
                        output_file.write(f"{pt_results[tf][tau]}\t")
                else:
                    output_file.write("N/A\t")
            output_file.write("\n")
        
        output_file.write("\n")
        
        # Write Data Transfer Time table
        output_file.write("Data Transfer Time (us)\n")
        output_file.write("-" * 30 + "\n")
        
        # Header row (tau values)
        output_file.write("tf\\tau\t")
        for tau in taus:
            output_file.write(f"{tau}\t")
        output_file.write("\n")
        
        # Data rows for data transfer time
        for tf in tfs:
            output_file.write(f"{tf}\t")
            for tau in taus:
                if dt_results[tf][tau] is not None:
                    if isinstance(dt_results[tf][tau], float):
                        output_file.write(f"{dt_results[tf][tau]:.2f}\t")
                    else:
                        output_file.write(f"{dt_results[tf][tau]}\t")
                else:
                    output_file.write("N/A\t")
            output_file.write("\n")
        
        output_file.write("\n" + "="*60 + "\n")
    
    print(f"Combined results saved to: {combined_filename}")

                # flag = False

# Notify the user where the outputs are saved
print(f"All results have been saved to the report/ directory")

