import torch
from torch import nn, Tensor
from typing import Callable, Optional
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from src import utils

class MLP(nn.Module):

    def __init__(self, 
        in_dim : int,
        hidden_dims : list[int],
        out_dim : int,
        hidden_act : Callable[[Tensor], Tensor] = F.relu,
        out_act : Callable[[Tensor], Tensor] = nn.Identity(),
        norm : bool = False
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.hidden_act = hidden_act
        self.out_act = out_act

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if norm else nn.Identity()
        ))

        for input_dim, output_dim in zip(hidden_dims[:-1], hidden_dims[1:]):

            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim) if norm else nn.Identity()
            ))
        self.out = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x : Tensor) -> Tensor:

        for layer in self.layers:
            x = self.hidden_act(layer(x))

        x = self.out(x)
        x = self.out_act(x)

        return x

class Recommender:

    def __init__(self,
        checkpoint_path : str,
        device : str           
    ) -> None:
        
        self.checkpoint_path = checkpoint_path
        self.device = device

        ### Load Checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        ### Load SentenceBERT Model
        self.sentence_bert = SentenceTransformer(self.checkpoint['sentence_bert_model'])

        ### Load Model
        self.model = MLP(
            in_dim=self.checkpoint['config']['in_dim'],
            hidden_dims=self.checkpoint['config']['hidden_dims'],
            out_dim=self.checkpoint['config']['out_dim'],
            hidden_act=getattr(F, self.checkpoint['config']['hidden_act']),
            out_act=getattr(F, self.checkpoint['config']['out_act']),
            norm=self.checkpoint['config']['norm']
        )

        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(device).eval()

        ### Venues List
        self.venues = self.checkpoint['venues']

    def get_venues(self,
        articles : list[str],
        batch_size : int,
        threshold : Optional[float] = None,
        top_k : Optional[int] = None,
    ) -> list[list[tuple[str, float]]]:
        
        if top_k is None and threshold is None:
            raise ValueError("threshold and top_k should not be simultaneously None.")
        
        ### Apply Sentence BERT
        articles_embeddings = torch.from_numpy(
            self.sentence_bert.encode(
                sentences=articles,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True,
                device=self.device
            )
        )

        ### Apply MLP-head
        loader = DataLoader(TensorDataset(articles_embeddings), batch_size=batch_size, shuffle=False)

        Y_hat = []

        for batch in loader:
            x = batch[0].to(self.device)
            y_hat = self.model.forward(x)
            y_hat = y_hat.detach().cpu()
            Y_hat.append(y_hat)

        Y_hat = torch.cat(Y_hat)

        ### Get venues
        row, col = torch.argwhere(Y_hat).t()
        probas = Y_hat[row, col]
        venues = [[] for _ in articles]

        for article_id, venue_id, proba in zip(row, col, probas):
            venues[article_id].append((self.venues[venue_id], proba))

        if top_k is not None:
            venues = [
                sorted(venue_list, key=lambda x : x[1], reverse=True)[:top_k]
                for venue_list in venues
            ]

        if threshold is not None:

            venues = [
                [
                    (venue, proba) for venue, proba in venue_list
                    if proba >= threshold
                ]
                for venue_list in venues
            ]

        return venues
    

__instances = {}

def load_model(chkp_path : str) -> Recommender:

    if chkp_path not in __instances:

        logger = utils.get_logger("main")

        device = utils.device()
        logger.info(f"Device {device}.")


        model = Recommender(chkp_path, device)
        __instances[chkp_path] = model

        logger.info("Model was loaded.")


    return __instances[chkp_path]