import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler, NeighborSampler
from utils.trick import gumbel_softmax


class PG_few(nn.Module):
    def __init__(self, feature, text_embedding, threshold: float, lamb, alpha, device="cuda"):
        super().__init__()
        self.feature = F.normalize(feature, p=2, dim=1)
        self.embedding = F.normalize(text_embedding, p=2, dim=1)
        self.threshold = threshold
        self.lamb = lamb
        self.device = device
        self.virtual = torch.nn.Parameter(torch.ones(1, feature.shape[1]))  # 64 * x
        self.alpha = alpha
        print("virtual", self.virtual.shape)
        torch.nn.init.kaiming_normal_(self.virtual)
        self.bce_loss = torch.nn.BCELoss()

    def s_diff(self, edges_adj, sim_sub):
        diff = torch.norm(edges_adj - sim_sub, 2, dim=1)
        diff = torch.mean(diff)
        return diff

    def f_diff(self, sim_sub: torch.Tensor, feature: torch.Tensor, embedding: torch.Tensor):
        emb_out = torch.sum(feature, dim=0) / feature.shape[0]
        emb_pg = torch.sum(embedding, dim=0) / sim_sub.shape[0]
        diff = torch.norm(emb_out - emb_pg, 2)
        return diff

    def forward(self, graph: dgl.DGLGraph, train_nodes: torch.Tensor, train_label: torch.Tensor):
        sampler = MultiLayerFullNeighborSampler(1)
        train_nodes = train_nodes.to(torch.int32)
        score = torch.zeros_like(train_nodes, dtype=torch.float32, device=self.device)  # train_nodes
        text_embedding_local = self.embedding - self.virtual
        text_embedding_local = F.normalize(text_embedding_local, p=2, dim=1)
        # print(self.virtual)
        for idx, node in enumerate(train_nodes):
            idx_batch, _, blocks = sampler.sample(graph, node)
            idx_batch = torch.sort(idx_batch)[0]
            text_embedding_batch = text_embedding_local[idx_batch]
            sim = text_embedding_batch @ text_embedding_batch.T
            sim_sub = sim - self.threshold
            sim_sub = gumbel_softmax(sim_sub)
            edges_ego = blocks[0].edges()
            edges_ego = torch.stack(edges_ego, dim=0)
            n = idx_batch.shape[0]
            edges_adj = torch.zeros((n, n), device=self.device)
            edges_adj[edges_ego[0], edges_ego[1]] = 1
            s_diff = self.s_diff(edges_adj, sim_sub)
            f_diff = self.f_diff(sim_sub, self.feature[idx_batch], self.embedding[idx_batch])
            score[idx] = s_diff + self.alpha * f_diff
        # print(score, train_label, score.shape, train_l abel.shape)
        # loss = torch.mean(score) + self.bce_loss(score[train_idx], train_label.float())
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        loss = self.bce_loss(score, train_label.float()) / train_label.shape[0]
        loss = loss.mean()
        return score, loss

    def inference(self, graph: dgl.DGLGraph, nodes: torch.Tensor):
        sampler = MultiLayerFullNeighborSampler(1)
        score = torch.zeros_like(nodes, dtype=torch.float32, device=self.device)
        text_embedding_local = self.embedding - self.virtual
        text_embedding_local = F.normalize(text_embedding_local, p=2, dim=1)

        for idx, node in enumerate(nodes):
            idx_batch, _, blocks = sampler.sample(graph, node)
            idx_batch = torch.sort(idx_batch)[0]
            text_embedding_batch = text_embedding_local[idx_batch]
            sim = text_embedding_batch @ text_embedding_batch.T
            sim_sub = sim - self.threshold
            sim_sub = gumbel_softmax(sim_sub)
            edges_ego = blocks[0].edges()
            edges_ego = torch.stack(edges_ego, dim=0)
            n = idx_batch.shape[0]
            edges_adj = torch.zeros((n, n), device=self.device)
            edges_adj[edges_ego[0], edges_ego[1]] = 1
            s_diff = self.s_diff(edges_adj, sim_sub)
            f_diff = self.f_diff(sim_sub, self.feature[idx_batch], self.embedding[idx_batch])
            score[idx] = s_diff + self.alpha * f_diff

        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        return score
