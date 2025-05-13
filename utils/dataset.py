import torch




def train_test(label: torch.Tensor, few_shot_num=5, device="cuda"):
    ones_idx = torch.nonzero(label == 1)  # 获取所有1的位置
    zeros_idx = torch.nonzero(label == 0)  # 获取所有0的位置
    idx_all = torch.arange(label.shape[0], device=device)
    # 随机选择5个1和5个0
    selected_ones_idx = ones_idx[torch.randperm(ones_idx.size(0))[:few_shot_num]]  # 随机选择5个1
    selected_zeros_idx = zeros_idx[torch.randperm(zeros_idx.size(0))[:few_shot_num]]  # 随机选择5个0
    # 构建训练集索引
    train_idx = torch.cat([selected_ones_idx, selected_zeros_idx])
    # 构建测试集索引（剩余的1和0）
    test_idx = idx_all[~torch.isin(label, train_idx)]
    # 构建训练集和测试集
    train_set = label[train_idx]
    test_set = label[test_idx]
    return train_idx, train_set, test_idx, test_set
