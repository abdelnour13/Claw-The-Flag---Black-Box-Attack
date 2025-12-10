import os
import zipfile
import shutil
import asyncio
import uuid
import tempfile
import json
from fastapi import FastAPI, UploadFile, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Tuple
from contextlib import asynccontextmanager
from sqlmodel import select, func
from sqlalchemy.orm import aliased
from dataclasses import asdict
from .models import SubmitResponse, StartJobResponse, LeaderboardSubmission
from .db import create_db_and_tables, SessionDep, Submit
from .job import Job, run_submission
from .utils import parse_file_size
from .constants import MAX_REQUEST_SIZE

### Run Job function
async def run_job() -> None:

    while True:

        job, session = await job_queue.get()
        
        if job is not None:

            ### Run Sumbission
            res = run_submission(job)

            ### Insert to database
            submission = Submit(
                id=res.job_id,
                team=job.team,
                score=res.score,
                status="SUCCESS" if res.success else "FAILED",
                reason=res.reason
            )

            session.add(submission)
            session.commit()

            ### Emit event
            job_events[job.job_id] = json.dumps(asdict(res))

        job_queue.task_done()

### Database and tables creation
@asynccontextmanager
async def lifespan(app : FastAPI):
    create_db_and_tables()
    asyncio.create_task(run_job())
    yield

### Create App
app = FastAPI(lifespan=lifespan)
public_router = APIRouter()
internal_router = APIRouter()

### Memory
jobs : Dict[str, Job] = {}
job_events = {}
job_queue : asyncio.Queue[Tuple[Job, SessionDep]] = asyncio.Queue()

### Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Mount Static Folder
app.mount("/public", StaticFiles(directory="public"), name="static")


@public_router.get("/")
def main_page():
    return FileResponse('public/index.html')

@public_router.get("/leaderboard")
def leaderboard():
    return FileResponse('public/leaderboard.html')

@public_router.get("/details")
def details():
    return FileResponse('public/details.html')

@public_router.get("/429")
def error_page_429():
    return FileResponse('public/429.html')


@internal_router.post("/submit")
def submit(file : UploadFile) -> SubmitResponse:

    ### Check File Size
    
    if file.size >= parse_file_size(MAX_REQUEST_SIZE):
        return SubmitResponse(success=False,reason=f"File is too large, maximum is : {MAX_REQUEST_SIZE}.")

    ### check if it's a valid zip file
    if not zipfile.is_zipfile(file.file):
        return SubmitResponse(success=False,reason="Not a valid zip file.")
        
    try:

        submits = Path("submits")

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            ### Extract File
            with zipfile.ZipFile(file.file, 'r') as f:
                f.extractall(temp_dir)

            dirname : Path = temp_dir / os.path.splitext(file.filename)[0]

            ### check if directory contains main.py
            main_file : Path = dirname / 'main.py'

            if not main_file.exists():
                return SubmitResponse(success=False, reason="Main file doesn't exist.")

            ### Copy resources to the job
            resources = Path("resources")

            keywords_src = resources / 'keywords.txt'
            articles_src = resources / 'articles.csv'

            keywords_dst = dirname / 'keywords.txt'
            articles_dst = dirname / 'articles.csv'

            shutil.copyfile(keywords_src, keywords_dst)
            shutil.copyfile(articles_src, articles_dst)

            ### Save directory
            job_id = str(uuid.uuid4())
            job_dir = str(submits / job_id)
            shutil.copytree(dirname, job_dir)

        ### Save The Job
        jobs[job_id] = Job(
            job_id=job_id,
            team=os.path.splitext(file.filename)[0],
            filename=job_dir
        )

    finally:
        file.file.close()

    return SubmitResponse(job_id=job_id,success=True)

@internal_router.post("/start-job/{job_id}")
async def start_job(job_id : str, session: SessionDep) -> StartJobResponse:

    job = jobs.get(job_id)

    if job is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail={
                "job_id" : job_id,
                "message" : f"Job with id {job_id} doesn't exist."
            }
        )
    
    await job_queue.put((job,session))
    jobs.pop(job_id, None)
    return StartJobResponse(success=True)

@internal_router.get("/events/{job_id}")
async def sse(job_id: str) -> StreamingResponse:

    async def generator() -> AsyncGenerator[str, None]:

        while True:

            if job_id in job_events:
                event = job_events.pop(job_id)
                yield f"data: {event}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")

@internal_router.get("/leaderboard-data")
def leaderboard_data(session : SessionDep) -> List[LeaderboardSubmission]:

    SubmitInner = aliased(Submit)

    subq = (
        select(SubmitInner.id)
        .where(
            SubmitInner.team == Submit.team,
            SubmitInner.status == "SUCCESS"
        )
        .order_by(SubmitInner.score.desc())
        .limit(1)
        .correlate(Submit)
        .scalar_subquery()
    )

    rank_col = func.rank().over(order_by=Submit.score.desc())

    query = (
        select(Submit, rank_col.label("rank"))
        .where(Submit.id == subq)
        .order_by("rank")
    )

    results = session.exec(query).all()

    return [
        LeaderboardSubmission(
            id=result.id, 
            team=result.team,
            score=result.score,
            status=result.status,
            reason=result.reason,
            rank=rank
        )
        for result, rank in results
    ]

app.include_router(public_router, prefix="/exposed")
app.include_router(internal_router, prefix="/internal")