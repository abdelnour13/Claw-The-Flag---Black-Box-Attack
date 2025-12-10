import os
from sqlmodel import Field, Session, SQLModel, create_engine
from typing import Optional, Annotated
from fastapi import Depends

### Model
class Submit(SQLModel, table=True):
    id : str = Field(primary_key=True)
    team : str = Field()
    score : Optional[float] = Field()
    status : str = Field()
    reason : Optional[str] = Field()

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