#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import queue
import threading


# 실행할 matrix 파일 목록 (필요에 맞게 수정)
MATRIX_LIST = [
    # "af23560",
    # "t2dah_a",
    # "raefsky1",
    # "nasa2910",
    "crystk01",
    # "bcsstk24",
    # "ex9",
    # "s3rmt3m3",
    # "cavity26",
    # "poisson3Da"    
]
# tfm_list = []
tau_m_list = [2,4,8,16,32]
bank_num = 48

def run_one(script: str, matrix_path: str, tfm: int, tfn: int, tau_m: int, tau_n: int, bank:int, out_folder: str, log_file: str) -> int:
    command = [
        "python3",
        script,
        "-mat", matrix_path,
        "-Tfm", str(tfm),
        "-Tfn", str(tfn),
        "-tau_m", str(tau_m),
        "-tau_n", str(tau_n),
        "-b", str(bank),
        "-o", out_folder,
    ]

    print(f"\n[RUN] {' '.join(command)}")
    with open(log_file, "w", encoding="utf-8") as log_f:
        result = subprocess.run(command, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"[FAIL] matrix={matrix_path}, returncode={result.returncode}, log={log_file}")
    else:
        print(f"[DONE] matrix={matrix_path}, log={log_file}")
    return result.returncode


def worker(job_queue: queue.Queue, args, fail_counter, lock):
    while True:
        try:
            matrix, bank = job_queue.get_nowait()
        except queue.Empty:
            return

        matrix_out = os.path.join(args.out_folder, f"{matrix}")
        os.makedirs(matrix_out, exist_ok=True)
        log_file = os.path.join(
            args.log_dir,
            f"{matrix}-{args.tau_m*args.tau_n*2}-{args.Tfm*args.Tfn}-b{bank}.log"
        )

        rc = run_one(
            script="final_gp_product.py",
            # script="final_morereal.py",
            matrix_path=f"/Data4/home/97ms_local/mat/{matrix}/{matrix}.mtx",
            # matrix_path=f"/mnt/data4/home/97ms_local/mat/{matrix}/{matrix}.mtx",
            tfm=args.Tfm,
            tfn=args.Tfn,
            # tau_m=tau_m,
            # tau_n=args.tau_n,
            tau_m=args.tau_m,
            tau_n=args.tau_n,
            bank=bank,
            out_folder=matrix_out,
            log_file=log_file,
        )

        if rc != 0:
            with lock:
                fail_counter[0] += 1

        job_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Run final_gp_product.py for a predefined matrix list")
    # parser.add_argument("--script", type=str, default="final_gp_product.py", help="Simulator script path")
    parser.add_argument("-Tfm", type=int, default=8, help="Tiling factor m")
    parser.add_argument("-Tfn", type=int, default=6, help="Tiling factor n")
    parser.add_argument("-tau_m", type=int, default=2, help="tau_m")
    parser.add_argument("-tau_n", type=int, default=1024, help="tau_n")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("-o", "--out-folder", type=str, default="trace_matrix_list", help="Output folder")
    args = parser.parse_args()

    if args.workers < 1:
        print("workers must be >= 1")
        sys.exit(1)

    os.makedirs(args.out_folder, exist_ok=True)
    log_dir = os.path.join(args.out_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    total_jobs = len(MATRIX_LIST) * bank_num
    fail_counter = [0]
    lock = threading.Lock()
    job_queue = queue.Queue()

    for matrix in MATRIX_LIST:
        for b in range(bank_num):
            job_queue.put((matrix, b))

    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(job_queue, args, fail_counter, lock), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    fail_count = fail_counter[0]

    print("\n====================")
    print(f"Total jobs : {total_jobs}")
    print(f"Failed     : {fail_count}")
    print(f"Succeeded  : {total_jobs - fail_count}")
    print("====================")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
