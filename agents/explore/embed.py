# refer to https://arxiv.org/abs/2008.02790, https://github.com/ezliu/dream
from typing import List
import torch
from torch.nn import functional as F
from torch import nn


class ProblemIDEmbedder(nn.Module):
    '''
    \mu -> z
    '''

    def __init__(self, num_train_problem, embed_dim):
        super(ProblemIDEmbedder, self).__init__()

        self.embed = nn.Embedding(num_train_problem, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.final_layer = nn.Linear(128, embed_dim)

    def forward(self, idices):
        hidden = self.hidden_layer(F.relu(self.embed(idices)))
        return self.final_layer(F.relu(hidden))

class ProblemHandler():
    def __init__(self, num_train_game):
        super(ProblemHandler, self).__init__()
        self.problem_id_embedder = ProblemIDEmbedder(num_train_game, 64).cuda()
        self.optimizer = torch.optim.Adam(self.problem_id_embedder.parameters(), lr=0.001)

    def get_problem_embeddings(self, problem_ids: List[int]):
        problem_ids = torch.tensor(problem_ids).long().cuda()
        return self.problem_id_embedder(problem_ids)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.problem_id_embedder.zero_grad()
        self.optimizer.zero_grad()     

    def parameters(self):
        return self.problem_id_embedder.parameters()   

    def save_model_to_path(self, save_to):
        torch.save(self.problem_id_embedder.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))