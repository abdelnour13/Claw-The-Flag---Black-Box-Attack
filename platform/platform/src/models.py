from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class SubmitResponse(BaseModel):
    success : bool
    job_id : Optional[str] = None
    reason : Optional[str] = None

class StartJobResponse(BaseModel):
    success : bool

class LeaderboardSubmission(BaseModel):
    id : str
    team : str
    score : Optional[float]
    status : str
    reason : Optional[str]
    rank : int
    created_at : str