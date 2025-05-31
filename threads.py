import os
my_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
total_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
