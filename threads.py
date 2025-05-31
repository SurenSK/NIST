import os
from time import sleep

my_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
total_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

print(f"Seeing my_id={my_id}, total_count={total_count}")

sleep(10)