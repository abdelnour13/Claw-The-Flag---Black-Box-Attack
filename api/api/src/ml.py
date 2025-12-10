import torch
import torch
from torch import nn, Tensor
from typing import Callable, Tuple, Optional
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
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
                nn.BatchNorm1d(out_dim) if norm else nn.Identity()
            ))


        self.out = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x : Tensor) -> Tensor:

        for layer in self.layers:
            x = self.hidden_act(layer(x))

        x = self.out(x)
        x = self.out_act(x)

        return x

class Model(nn.Module):

    def __init__(self,
        in_dim : int,
        hidden_dims : list[int],
        out_dim : int,
        act : Callable[[Tensor], Tensor] = F.relu,
        norm : bool = False
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.act = act,

        self.l_mlp = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            hidden_act=act,
            out_act=nn.Identity(),
            norm=norm
        ) 

        self.r_mlp = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            hidden_act=act,
            out_act=nn.Identity(),
            norm=norm
        ) 

        self.t = nn.Parameter(torch.scalar_tensor(1.0))

    def forward(self, 
        l : Tensor, 
        r : Tensor,
    ) -> Tuple[Tensor, Tensor]:

        l = self.l_mlp(l)
        r = self.r_mlp(r)

        return l, r


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
        self.model = Model(
            in_dim=self.checkpoint['config']['in_dim'],
            hidden_dims=self.checkpoint['config']['hidden_dims'],
            out_dim=self.checkpoint['config']['out_dim'],
            act=getattr(F, self.checkpoint['config']['act']),
            norm=self.checkpoint['config']['norm']
        )

        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(device).eval()

        ### Keyword Embeddings
        self.K : Tensor = self.checkpoint['K'].to(self.device)

        ### Keywords List
        self.keywords_list = self.checkpoint['keywords']

    @torch.no_grad()
    def get_keywords(self,
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

        ### Apply MLP-tuned
        loader = DataLoader(TensorDataset(articles_embeddings), batch_size=batch_size, shuffle=False)

        articles_embeddings = []

        for batch in loader:
            x = batch[0].to(self.device)
            x = self.model.l_mlp.forward(x)
            x = x.detach().cpu()
            articles_embeddings.append(x)

        articles_embeddings = torch.cat(articles_embeddings)

        threshold = threshold or 0.0
        loader = DataLoader(TensorDataset(articles_embeddings), batch_size=batch_size, shuffle=False)
        keywords = [[] for _ in range(len(articles))]
        offset = 0

        for batch in loader:

            x = batch[0].to(self.device)
            scores = torch.sigmoid((x @ self.K.T))

            threshold = threshold or 0.0

            if top_k:
                k_th = torch.sort(scores, -1, descending=True).values[:,:top_k][:,[-1]]
                thresholds = torch.where(threshold <= k_th, k_th, threshold)
            
            article_ids, keyword_ids = torch.argwhere(scores >= thresholds).t().cpu()

            for article_id, keyword_id in zip(article_ids.tolist(), keyword_ids.tolist()):

                score = scores[article_id, keyword_id].item()

                keywords[article_id + offset].append((
                    self.keywords_list[keyword_id],
                    score
                ))

                keywords[article_id + offset] = sorted(keywords[article_id + offset], key=lambda x : x[1], reverse=True)
            
            offset += len(x)

        return keywords
    

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