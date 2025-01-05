import torch
from torch import nn
import math

class LoraLayer(nn.Module):
    def __init__(self, raw_linear, r, alpha):
        super(LoraLayer, self).__init__()
        self.in_features = raw_linear.in_features
        self.out_features = raw_linear.out_features
        self.r = r
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((self.in_features, r)))
        self.lora_b = nn.Parameter(torch.zeros((r, self.out_features)))
        nn.init.kaiming_uniform_(self.lora_a, math.sqrt(5))

    def forward(self, x):  # x:(batch size,in features)
        device = torch.device("cuda")
        x = x.to(device)
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.r)
        return lora_output

