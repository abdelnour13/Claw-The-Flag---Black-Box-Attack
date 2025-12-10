import subprocess
import os
import numpy as np
import shutil
import dotenv
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from .score import soft_jackidx_score
from .constants import WORKER_MEMORY, WORKER_SWAP_MEMORY, WORKER_CPUS, WORKER_USE_GPU, WORKER_TIMEOUT

@dataclass
class Job:
    job_id : str
    team : str
    filename : str

@dataclass
class Response:
    job_id : str
    success : bool
    reason : Optional[str]
    score : Optional[float]

def run_submission(job : Job) -> Response:

    ### Response
    res = Response(
        job_id=job.job_id,
        success=True,
        reason=None,
        score=None
    )
    
    ### Run main file
    try:

        submits = os.environ['SUBMITS_HOST_PATH']

        cmd = filter(None, [
            "docker", 
            "run", 
            "--rm",
            "--memory", WORKER_MEMORY,
            "--memory-swap", WORKER_SWAP_MEMORY,
            "--cpus", WORKER_CPUS, 
            ("--gpus all" if WORKER_USE_GPU else None), 
            "--network", "none",               
            "-v", f"{os.path.join(submits, job.job_id)}:/job:rw",        
            "job-worker:latest",
            "bash", "-c", 'cd /job && python main.py'
        ])

        subprocess.run(
            cmd,
            cwd=job.filename,
            check=True,
            timeout=WORKER_TIMEOUT
        )

    except subprocess.CalledProcessError as e:
        res.success = False
        res.reason = f"Job failed due to an error happened during its excution."
    except subprocess.TimeoutExpired as e:
        res.success = False
        res.reason = f"Job failed because it took too long, limit is {dotenv.get_key(".env", "WORKER_TIMEOUT")}s."

    ### Check if job produced scores.npy
    if res.success:
        
        results_file = os.path.join(job.filename, "scores.npy")

        if not os.path.exists(results_file):
            res.success = False
            res.reason = "Job produced no scores.npy file."
    
    ### Check if it is a numpy file
    if res.success:

        try:
            preds = np.load(results_file, allow_pickle=False)
        except:
            res.success = False
            res.reason = "Job produced an invalid numpy file."
    
    if res.success:

        ### Load Target
        resources = Path('resources')
        target_file = resources / "scores.npy"
        target = np.load(target_file, allow_pickle=False)

        ### Check Dimensions
        if preds.shape != target.shape:
            res.success = False
            res.reason = f"Prediction dimension {preds.shape} didn't match target dimension {target.shape}"
        else: ### Compute Score
            res.score = soft_jackidx_score(preds, target)

    ### Delete directory
    shutil.rmtree(job.filename)

    ### Return result
    return res