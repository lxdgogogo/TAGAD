import argparse
import datetime
import os
import random
import sys
import time

import dgl
import torch
from torch.utils.data import DataLoader

os.chdir(sys.path[0])
sys.path.append("..")
from utils.load_data import load_data
from Local.PG_zero import PG
from utils.eval_utils import eval_model
from Global.anomaly_clip import AnomalyClip


def main(args):
    dropout = args.dropout
    batch_size = args.batch_size  # 64
    alpha = args.alpha
    norm = "batchnorm"
    activation = "prelu"
    dataset = args.dataset  # cora arxiv cora pubmed
    epsilon = args.epsilon  # 0.98
    lamb = args.lamb  # 0.5
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    random.seed(42)
    start = time.time()
    graph = load_data(dataset, device)
    now_date = datetime.date.today().strftime("%Y-%m-%d")
    text_emb = graph.ndata['text_embedding'].to(device)
    last_hidden_size = text_emb.shape[1]
    print("node sum: ", graph.number_of_nodes())
    feature = graph.ndata["feature"].float()
    pretrain = AnomalyClip(feature.shape[1], last_hidden_size, feature.shape[1], 1, dropout, norm, activation, device).to(device)
    pretrain.load_state_dict(torch.load(f'../model/{dataset}_{now_date}.pth'))
    pretrain.eval()
    with torch.no_grad():
        label_pre, _ = pretrain(graph, text_emb, feature, False, batch_size)
        emb = pretrain.text_projection(text_emb)
        pg = PG(feature, emb, epsilon, alpha, device=device)
        score = pg.cal_score(graph)
    print(torch.max(score), torch.min(score), torch.max(label_pre), torch.min(label_pre))
    score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
    final_pre = label_pre + lamb * score
    final_pre = (final_pre - torch.min(final_pre)) / (torch.max(final_pre) - torch.min(final_pre))
    end = time.time()
    eval_model(graph.ndata['label'].cpu().numpy(), final_pre.cpu().numpy(), str(end - start), f"{args.file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='arxiv')  # cora arxiv pubmed
    parser.add_argument('--hidden_size', type=int, default=1024)  # 1024
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--epsilon', type=float, default=0.98)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.7)  # 0.3
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--file_name', type=str, default="")
    args = parser.parse_args()
    main(args)
