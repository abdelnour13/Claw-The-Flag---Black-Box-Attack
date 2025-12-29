import sys
sys.path.append("../../..")
import os
import requests
import string
import re
import utils
import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm.auto import tqdm
from lingua import LanguageDetectorBuilder, Language
from argparse import ArgumentParser

DEFAULT_ATTRIBUTES = [
    "docid",
    "title_s",
    "keyword_s",
    "abstract_s",
    "authFullName_s",
    "producedDate_tdate"
    "domainAllCode_s"
]

@dataclass
class Query:
    teams : list[str] = field(default_factory = lambda : ["LS2N-DUKE"])
    format : str = "json"
    attributes : list[str] = field(default_factory = lambda : DEFAULT_ATTRIBUTES)
    from_ : str = "2011-08-12T20:17:46.384Z"
    to : str = "*"
    languages : list[str] = field(default_factory = lambda : ["en"])
    sort : str = "producedDate_tdate desc"

@dataclass
class DataDownloaderConfig:
    query : Query = field(default_factory = Query)
    num_row_per_page : int = 32
    num_pages : Optional[int] = None

@dataclass
class Args:
    config : str = "default"
    output : str = "data"

class DataDownloader:

    BASE_URL : str = "https://api.archives-ouvertes.fr/search"

    def __init__(self, config : DataDownloaderConfig) -> None:

        self.config = config

        self.detector = LanguageDetectorBuilder \
            .from_languages(Language.ENGLISH) \
            .build()

    def normalize_keyword(self, keyword : str) -> str:
        translator = str.maketrans('', '', string.punctuation)
        keyword = keyword.lower()
        keyword = keyword.replace('-',' ')
        keyword = re.sub('[0-9]+','',keyword)
        keyword = keyword.translate(translator)
        return keyword

    def process_doc(self, doc : dict) -> Optional[dict]:

        processed_doc = dict()

        processed_doc["id"] = doc["docid"]
        processed_doc["title"] = None
        processed_doc['abstract'] = None
        processed_doc["keywords"] = doc.get("keyword_s", [])
        processed_doc["authors"] = doc.get("authFullName_s", [])
        processed_doc["venue"] = doc.get("domainAllCode_s", [])
        processed_doc["date"] = doc["producedDate_tdate"]

        processed_doc["keywords"] = list(map(self.normalize_keyword, processed_doc["keywords"]))

        for title in doc["title_s"]:

            try:
                if self.is_english(title):
                    processed_doc["title"] = title
                    break
            except:
                continue
        else:
            return None
        
        for abstract in doc["abstract_s"]:

            try:
                if self.is_english(abstract):
                    processed_doc["abstract"] = abstract
                    break
            except:
                continue
        else:
            return None

        return processed_doc
    
    def is_english(self, text : str) -> bool:
        return self.detector.detect_language_of(text) == Language.ENGLISH
    
    def get_url(self, 
        team : str,
        page_id : int = 0
    ) -> str:

        return ''.join([
            DataDownloader.BASE_URL,
            f'/{team}/',
            f'?wt={self.config.query.format}',
            f'&fl={",".join(self.config.query.attributes)}',
            f'&fq=producedDate_tdate:[{self.config.query.from_} TO {self.config.query.to}]',
            f'&fq=language_s:({",".join(self.config.query.languages)})',
            f'&sort={self.config.query.sort}',
            f'&rows={self.config.num_row_per_page}',
            f'&start={page_id * self.config.num_row_per_page}',
            f'&fq=abstract_s:["" TO *]',
        ])
    
    def download(self, output_path : str):

        num_pages = self.config.num_pages

        for team in self.config.query.teams:

            if self.config.num_pages is None:

                try:
                    response = requests.get(self.get_url(team))
                    data = response.json()
                    num_docs = data["response"]["numFound"]
                    num_pages = num_docs // self.config.num_row_per_page + 1
                except Exception as e:
                    print(e)
                    num_docs = 0
                    num_pages = 0
        
            for page_id in tqdm(range(389, num_pages), desc = f"Downloading documents for team : {team}"):
                
                url = self.get_url(team, page_id)

                try:
                    
                    response = requests.get(url)
                    data = response.json()
                    data = data["response"]["docs"]

                    data = list(
                        map(
                            json.dumps,
                            filter(
                                lambda x : x is not None,
                                map(self.process_doc, data)
                            )
                        )
                    )

                    content = "\n".join(data) + "\n"

                    with open(output_path, "a") as f:
                        f.write(content)
                    
                except Exception as e:
                    print(e)
                    break
    
def main(args : Args) -> None:

    config_path = f'{args.config}.json'
    output_path = os.path.join("data", f'{args.output}.raw.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config = utils.load_dataclass(DataDownloaderConfig, config_path)

    data_downloader = DataDownloader(config)
    docs = data_downloader.download(output_path)
    
if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default=Args.config)
    parser.add_argument('--output', type=str, default=Args.output)

    args = Args(**vars(parser.parse_args()))

    main(args)