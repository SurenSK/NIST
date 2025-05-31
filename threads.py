import os
import subprocess
from time import sleep

result = subprocess.run(['squeue', '-u', 'skumar43', '-h', '-t', 'R'], capture_output=True, text=True)
total_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

my_id = int(os.environ['SLURM_JOBID'])

print(f"Seeing my_id={my_id}, total_count={total_count}")
sleep(10)