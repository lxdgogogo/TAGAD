import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler, NeighborSampler


class PG:
    def __init__(self, feature, text_embedding, epsilon: float, alpha, walk=-1, device="cuda"):
        self.feature = F.normalize(feature, p=2, dim=1)
        norm_x = torch.norm(text_embedding, p=2, dim=1, keepdim=True)
        text_embedding = text_embedding / norm_x
        # sim = text_embedding @ text_embedding.T
        self.embedding = text_embedding
        # self.sim = (sim >= epsilon).to(torch.int)
        self.epsilon = epsilon
        self.alpha = alpha
        self.walk = walk
        self.device = device

    def s_diff(self, ego_adj, sim_sub):
        diff = ego_adj - sim_sub
        diff = torch.norm(diff, 2, dim=1)
        # print(prompt_graph.number_of_edges(), ego_graph.number_of_edges())
        diff = torch.mean(diff)
        return diff

    def f_diff(self, feature: torch.Tensor, embedding: torch.Tensor):
        emb_out = torch.sum(feature, dim=0) / feature.shape[0]
        emb_pg = torch.mean(embedding, dim=0)
        diff = torch.norm(emb_out - emb_pg, 2)
        return diff

    def cal_score(self, graph: dgl.DGLGraph):
        score = torch.empty(graph.num_nodes(), device=self.device)
        if self.walk == -1:
            sampler = MultiLayerFullNeighborSampler(1)
        else:
            sampler = NeighborSampler([self.walk])
        # sampler = MultiLayerFullNeighborSampler(1)
        for node in graph.nodes():
            idx, _, blocks = sampler.sample(graph, node)
            idx_batch = torch.sort(idx)[0].to(self.device)
            sim = self.embedding[idx_batch] @ self.embedding[idx_batch].T
            sim = (sim >= self.epsilon).to(torch.int)
            # ego_graph = dgl.node_subgraph(graph, idx).to(self.device)
            edges_ego = blocks[0].edges()
            edges_ego = torch.stack(edges_ego, dim=0)
            n = idx_batch.shape[0]
            edges_adj = torch.zeros((n, n), device=self.device)
            edges_adj[edges_ego[0], edges_ego[1]] = 1
            s_diff = self.s_diff(edges_adj, sim)
            f_diff = self.f_diff(self.feature[idx_batch], self.embedding[idx_batch])
            score[node] = s_diff + self.alpha * f_diff
            # score[node] = s_diff
            torch.cuda.empty_cache()
        return score
