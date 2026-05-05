
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

# tfs = [1,2,4,8,16,32]
tfs = [32,64,128,256,512,1024]
taus = [32,64,128,256,512,1024]
mats = ["nasa2910","raefsky1","ex9","bcsstk24","cavity26","crystk01","s3rmt3m3","t2dah_a","poisson3Da","af23560"]


# tfs = [2]
# taus = [32]
# mats = ["nasa2910"]



# MATRIX_PATH = "/Data4/home/97ms_local/mat"
MATRIX_PATH = "mat"

# Create report directory if it doesn't exist
os.makedirs("report", exist_ok=True)

# Function to process a single matrix
def process_matrix(mat):
    print(f"[Thread] Processing matrix: {mat}")
    
    # Create dictionaries to store results for this matrix
    pt_results = {}  # Partitioning time results
    dt_results = {}  # Data transfer time results
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
            print(f"[{mat}] Running: tf={tf}, tau={tau}")
            
            # Prepare the command
            command = ["time", "./build/X86/gem5.opt", 
                       "configs/deva2.py",
                       "-mat", str(MATRIX_PATH+"/"+mat+".mtx"),
                       '-tau', str(tau), '-Tf', str(tf)
                       ]
            print(f"[{mat}] Running command: {' '.join(command)}")

            # Execute the command and capture output
            try:
                # Create log filename for this specific run
                log_filename = f"report/v2/{mat}_tf{tf}_tau{tau}_log.txt"
                
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                    pt = None
                    dt = None
                    log_content = []

                    # Open log file for writing
                    with open(log_filename, "w") as log_file:
                        # Write command info to log
                        log_file.write(f"Command: {' '.join(command)}\n")
                        log_file.write(f"Matrix: {mat}, tf: {tf}, tau: {tau}\n")
                        log_file.write("="*60 + "\n\n")
                        
                        for line in proc.stdout:
                            # Write to log file
                            log_file.write(line)
                            log_file.flush()  # Ensure immediate writing
                            
                            # Also print to console with matrix identifier
                            print(f"[{mat}] {line}", end='')
                            
                            # Store in memory for processing
                            log_content.append(line)

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

                    # Log completion status
                    with open(log_filename, "a") as log_file:
                        log_file.write(f"\n{'='*60}\n")
                        if proc.returncode == 0:
                            log_file.write(f"Command executed successfully\n")
                            print(f"[{mat}] Command executed successfully")
                        else:
                            log_file.write(f"Command failed with return code {proc.returncode}\n")
                            print(f"[{mat}] Command failed with return code {proc.returncode}")
                        
                        log_file.write(f"Log saved to: {log_filename}\n")
                    
                    print(f"[{mat}] Log saved to: {log_filename}")

            except subprocess.CalledProcessError as e:
                print(f"[{mat}] Error executing command")
                print(f"[{mat}] Error:", e)
                pt_results[tf][tau] = "ERROR"
                dt_results[tf][tau] = "ERROR"
                
                # Save error to log file
                error_log_filename = f"report/v2/{mat}_tf{tf}_tau{tau}_error_log.txt"
                with open(error_log_filename, "w") as error_log:
                    error_log.write(f"Command: {' '.join(command)}\n")
                    error_log.write(f"Matrix: {mat}, tf: {tf}, tau: {tau}\n")
                    error_log.write("="*60 + "\n\n")
                    error_log.write(f"Error: {e}\n")
                print(f"[{mat}] Error log saved to: {error_log_filename}")

    # Write combined file with both partitioning time and data transfer time
    combined_filename = f"report/v2/{mat}_results.txt"
    
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
                        output_file.write(f"{pt_results[tf][tau]:.4f}\t")
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
                        output_file.write(f"{dt_results[tf][tau]:.4f}\t")
                    else:
                        output_file.write(f"{dt_results[tf][tau]}\t")
                else:
                    output_file.write("N/A\t")
            output_file.write("\n")
        
        output_file.write("\n" + "="*60 + "\n")
    
    print(f"[{mat}] Combined results saved to: {combined_filename}")
    return f"Matrix {mat} completed"

# Main execution with threading
if __name__ == "__main__":
    print(f"Starting processing of {len(mats)} matrices in parallel...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor to run matrices in parallel
    with ThreadPoolExecutor(max_workers=min(len(mats), 20)) as executor:  # Limit to 20 concurrent threads
        # Submit all matrix processing tasks
        futures = [executor.submit(process_matrix, mat) for mat in mats]
        
        # Wait for all tasks to complete and collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
                print(f"✓ {result}")
            except Exception as e:
                print(f"✗ Error processing matrix: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Notify the user where the outputs are saved
    print(f"\n{'='*60}")
    print(f"All results have been saved to the report/ directory")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Processed {len(mats)} matrices in parallel")
    print(f"{'='*60}")

