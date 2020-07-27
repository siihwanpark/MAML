import torch
import torch.nn as nn
import torch.nn.functional as F

from sinusoid import Sinusoid
from torch.utils.data import DataLoader

train_dataset = Sinusoid(k_shot=10, q_query=15, num_tasks=2000000)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=True)

for src_train, trg_train, src_test, trg_test in train_loader:
    break