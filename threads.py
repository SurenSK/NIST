import os
import subprocess
from time import time, sleep
import torch
from diffusers import DiffusionPipeline
from statistics import mean
from peft import LoraConfig, get_peft_model
from scipy.stats import zscore
import glob

def sync_weights(model, score, jobid):
    tmp, candidate = f"{jobid}.pt", "candidate.pt"
    cData = {'weights': model.state_dict(), 'score': score}
    
    if glob.glob("*.pt"):
        t0 = time()
        while not os.path.exists(candidate) and time() - t0 < 60: sleep(2)
        if os.path.exists(candidate):
            os.rename(candidate, tmp)
            nData = torch.load(tmp)
            if cData['score'] <= nData['score']:
                cData = nData
    
    torch.save(cData, tmp)
    os.rename(tmp, candidate)
    return cData['weights']

def z():
    return

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large").to("cuda")
lora_config = LoraConfig(
    r=16,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_alpha=32,
    lora_dropout=0.1
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

maxIters = 1000
rolloutWidth = 16

squeueRes = subprocess.run(['squeue', '-u', 'skumar43', '-h', '-t', 'R'], capture_output=True, text=True)
nWorkers = len(squeueRes.stdout.strip().split('\n')) if squeueRes.stdout.strip() else 0
jobid = int(os.environ['SLURM_JOBID'])
print(f"Seeing id={jobid}, count={nWorkers}")

for _ in range(maxIters):
    images = [pipe(prompt).images[0] for _ in range(rolloutWidth)]
    scores = getScores(images)
    advantages = zscore(scores)
    loss = mean(advantages*logprobs)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pipe.unet.load_state_dict(sync_weights(pipe.unet, mean(scores), jobid))