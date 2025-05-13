import argparse
import os
import sys
import time
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np

os.chdir(sys.path[0])
sys.path.append("..")
from utils.early_stop import EarlyStopping
from utils.functions import print_trainable_parameters
from Global.anomaly_clip import AnomalyClip
from utils.eval_utils import eval_model
from utils.load_data import load_data


def main(args):
    token_length = args.token_length
    batch_size = args.batch_size  # 64 400
    max_epoch = args.epoch  # 1000
    lr = args.lr
    weight_decay = args.weight_decay
    patience = args.patience
    dropout = args.dropout
    num_layers = args.num_layers
    norm = "batchnorm"
    activation = "prelu"
    dataset = args.dataset
    file_name = args.file_name
    if file_name == "":
        file_name = f"{dataset}_pretrain"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    # dataset = "arxiv"
    graph = load_data(dataset, device)
    feature = graph.ndata["feature"].float()
    print(device)
    embeddings = graph.ndata["text_embedding"].to(device)
    last_hidden_size = embeddings.shape[1]
    print(last_hidden_size)
    pretrain = AnomalyClip(feature.shape[1], last_hidden_size, feature.shape[1], num_layers, dropout, norm, activation,
                           device).to(device)
    # model train
    epoch_iter = tqdm(range(max_epoch))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pretrain.parameters()), lr, weight_decay=weight_decay)
    pretrain, optimizer = accelerator.prepare([pretrain, optimizer])
    print_trainable_parameters(pretrain)
    stopper = EarlyStopping(patience=patience, dataset=f"{dataset}")
    pre = True
    start = time.time()
    for epoch in epoch_iter:
        if epoch > (max_epoch / 10):
            pre = False
        with accelerator.accumulate(pretrain):
            score, loss = pretrain(graph, embeddings, feature, pre, batch_size)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        auc = roc_auc_score(graph.ndata["label"].detach().cpu().numpy(), score.detach().cpu().numpy())
        epoch_iter.set_description(f'Epoch {epoch}: train_loss: {loss}, auc: {auc}')
        # print(f'Epoch {epoch}: train_loss: {loss_sum}, auc: {auc}')
        if pre == False:
            stopper.step(loss, pretrain)
            if stopper.early_stop:
                break

    pretrain.eval()
    end = time.time()
    pretrain.eval()
    score, loss = pretrain(graph, embeddings, feature, pre, batch_size)
    eval_model(graph.ndata["label"].detach().cpu().numpy(), score.detach().cpu().numpy(), str(end - start), file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pubmed')  # cora arxiv pubmed
    parser.add_argument('--epoch', type=int, default=100)  # 300
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--token_length', type=int, default=128)  # 128 256
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--file_name', type=str, default="")
    args = parser.parse_args()
    main(args)
