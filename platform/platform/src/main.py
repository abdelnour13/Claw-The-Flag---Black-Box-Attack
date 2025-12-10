import os
import zipfile
import shutil
import asyncio
import uuid
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, HTTPException, status, APIRouter, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import AsyncGenerator, Dict, Tuple
from contextlib import asynccontextmanager
from sqlmodel import select, func
from sqlalchemy.orm import aliased
from dataclasses import asdict
from .models import SubmitResponse, StartJobResponse
from .db import create_db_and_tables, SessionDep, Submit
from .job import Job, run_submission
from .utils import parse_file_size
from .constants import MAX_REQUEST_SIZE, END_DATE

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
def challenge_ended():
    return datetime.now() > END_DATE

def check_challenge_ended():

    if challenge_ended():

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message" : "Challenge has ended! Good luck."
            }
        )

app = FastAPI(lifespan=lifespan)
public_router = APIRouter()
internal_router = APIRouter(dependencies=[Depends(check_challenge_ended)])
templates = Jinja2Templates(directory="templates")

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

### Ednpoints
@public_router.get("/")
def main_page(request : Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "challenge_ended" : challenge_ended()
        }
    )

@public_router.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(
    request : Request,
    session : SessionDep
):

    ### SQL-Query
    SubmitInner = aliased(Submit)

    subq = (
        select(SubmitInner.id)
        .where(
            SubmitInner.team == Submit.team,
            SubmitInner.status == "SUCCESS"
        )
        .order_by(
            SubmitInner.score.desc(),
            SubmitInner.created_at.asc()
        )
        .limit(1)
        .correlate(Submit)
        .scalar_subquery()
    )

    rank_col = func.rank().over(
        order_by=[
            Submit.score.desc(),
            Submit.created_at.asc()
        ]
    )

    query = (
        select(Submit, rank_col.label("rank"))
        .where(Submit.id == subq)
        .order_by("rank")
    )

    ### Excute Query
    results = session.exec(query).all()

    ### Context to pass
    entries = [
        {
            "id" : result.id, 
            "team" : result.team,
            "hour" : result.created_at.strftime("%H:%M:%S"),
            "score" : result.score,
            "status" : result.status,
            "reason" : result.reason,
            "rank" : rank
        }
        for result, rank in results
    ]

    return templates.TemplateResponse(
        request=request,
        name="leaderboard.html",
        context={
            "entries" : entries
        }
    )

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
        job_id = str(uuid.uuid4())
        submission = submits / job_id

        ### Extract File
        with zipfile.ZipFile(file.file, 'r') as f:
            f.extractall(submission)

        ### check if directory contains main.py
        main_file : Path = submission / 'main.py'

        if not main_file.exists():
            return SubmitResponse(success=False, reason="Main file doesn't exist.")

        ### Copy resources to the job
        resources = Path("resources")

        keywords_src = resources / 'keywords.txt'
        articles_src = resources / 'articles.csv'

        keywords_dst = submission / 'keywords.txt'
        articles_dst = submission / 'articles.csv'

        shutil.copyfile(keywords_src, keywords_dst)
        shutil.copyfile(articles_src, articles_dst)

        ### Save The Job
        jobs[job_id] = Job(
            job_id=job_id,
            team=os.path.splitext(file.filename)[0].upper(),
            filename=str(submission)
        )

    finally:
        file.file.close()

    return SubmitResponse(job_id=job_id,success=True)

@internal_router.post("/start-job/{job_id}")
async def start_job(job_id : str, session: SessionDep) -> StartJobResponse:

    job = jobs.pop(job_id, None)

    if job is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail={
                "job_id" : job_id,
                "message" : f"Job with id {job_id} doesn't exist."
            }
        )
    
    await job_queue.put((job,session))
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

app.include_router(public_router, prefix="/exposed")
app.include_router(internal_router, prefix="/internal")