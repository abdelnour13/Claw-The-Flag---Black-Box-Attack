import os
from datetime import datetime, timezone
from sqlmodel import Field, Session, SQLModel, create_engine
from sqlalchemy import Column, DateTime, func
from typing import Optional, Annotated
from fastapi import Depends
from functools import partial

utcnow = partial(datetime.now, tz=timezone.utc)

### Model
class Submit(SQLModel, table=True):
    id : str = Field(primary_key=True)
    team : str = Field()
    score : Optional[float] = Field()
    status : str = Field()
    reason : Optional[str] = Field()
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

### Database
sqlite_file_name = os.path.join("db", "database.db")
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {
    "check_same_thread": False
}

engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]