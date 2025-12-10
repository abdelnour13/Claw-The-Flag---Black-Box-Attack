import constants as C
from pydantic import BaseModel, Field, field_validator
from typing import List
from src.utils import check_num_tokens

class Document(BaseModel):
    description : str = Field(
        ...,
        min_length=1,
        description=(
            f"Text description. Must contain between "
            f"at least {C.MIN_TOKENS} and at most {C.MAX_TOKENS} tokens."
        )
    )

    @field_validator('description')
    def validate_description(cls, v : str) -> str:
        return check_num_tokens(document=v,min_tokens=C.MIN_TOKENS,max_token=C.MAX_TOKENS)

class DocumentsList(BaseModel):
    documents : List[Document] = Field(min_length=1, max_length=C.MAX_DOCUMENTS)

class Keyword(BaseModel):
    name : str
    rank : int