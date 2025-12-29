import json
import sys
sys.path.append('../..')
from dataclasses import is_dataclass, fields
from typing import TypeVar,List,Dict,Type,Union

DS = TypeVar('DS')

def load_dataclass_from_dict(cls : Type[DS], data : dict) -> DS:

    new_data = dict()

    for field in fields(cls):

        if field.name in data:
            
            if is_dataclass(field.type):
                new_data[field.name] = load_dataclass_from_dict(field.type, data[field.name])
            else:
                new_data[field.name] = data[field.name]

    return cls(**new_data)

def load_json_file(filename : str) -> Union[Dict,List[dict]]:
    
    with open(filename, 'r') as f:
        data = json.load(f)

    return data

def load_dataclass(cls : Type[DS], filename : str) -> DS:
    json_ = load_json_file(filename)
    return load_dataclass_from_dict(cls, json_)


def save_json(obj : object, filename : str) -> None:

    with open(filename, "w") as f:
        json.dump(obj, f, indent = 4)