import pandas as pd
import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Callable
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

### Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

### Load Articles
articles = pd.read_csv("./articles.csv")["abstract"].tolist()
print(f"There is {len(articles)} article.")

### Load The Model (Original)
sentence_transformer = SentenceTransformer("./checkpoint", local_files_only=True)

articles_embeddings = torch.from_numpy(
    sentence_transformer.encode(
        sentences=articles,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        device=DEVICE,
        normalize_embeddings=True,
    )
)

print(f"Dimensions : {articles_embeddings.size(0)}x{articles_embeddings.size(1)}.")

### Load Finetuned Head
class MLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        hidden_act: Callable[[Tensor], Tensor] = F.relu,
        out_act: Callable[[Tensor], Tensor] = nn.Identity(),
        norm: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.hidden_act = hidden_act
        self.out_act = out_act

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]) if norm else nn.Identity(),
            )
        )

        for input_dim, output_dim in zip(hidden_dims[:-1], hidden_dims[1:]):

            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim) if norm else nn.Identity(),
                )
            )

        self.out = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers:
            x = self.hidden_act(layer(x))

        x = self.out(x)
        x = self.out_act(x)

        return x


chkp = torch.load("./mlp_head_chkp.pt", weights_only=False, map_location='cpu')

### Create MLP & Load weights
mlp_head = MLP(
    in_dim=chkp["config"]["in_dim"],
    hidden_dims=chkp["config"]["hidden_dims"],
    out_dim=chkp["config"]["out_dim"],
    hidden_act=getattr(F, chkp["config"]["hidden_act"]),
    out_act=getattr(F, chkp["config"]["out_act"]),
    norm=chkp["config"]["norm"],
)

print(mlp_head.load_state_dict(chkp['state_dict']))

mlp_head = mlp_head.to(DEVICE).eval()

### Inference
loader = DataLoader(
    dataset=TensorDataset(articles_embeddings), 
    batch_size=BATCH_SIZE, 
    drop_last=False, 
    shuffle=False
)

with torch.inference_mode():

    Y = []

    for x in loader:
        x = x[0].to(DEVICE)
        y = mlp_head(x)
        Y.append(y)

    Y = torch.cat(Y)

print(f"Dimensions : {Y.size(0)}x{Y.size(1)}.")

### Save
preds = Y.detach().cpu().numpy()
np.save(file="./scores.npy", arr=preds, allow_pickle=False)