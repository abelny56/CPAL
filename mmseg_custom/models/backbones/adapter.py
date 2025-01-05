import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        groups = 4  
        self.adapter_down = nn.Conv2d(320, 8, kernel_size=1, groups=groups)
        self.adapter_up = nn.Conv2d(8, 320, kernel_size=1, groups=groups)
        self.adapter_mid = nn.Conv2d(8, 8, kernel_size=1, groups=groups)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, D1, D2 = x.shape  
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down).view(B, D1, D2, C)
        return x_up

class Bi_direct_adapter_1(nn.Module):
    def __init__(self, dim=2, input_size=640):
        super().__init__()
        groups = 4
        self.adapter_down_1 = nn.Conv2d(input_size, 8, kernel_size=1, groups=groups)
        self.adapter_up_1 = nn.Conv2d(8, input_size, kernel_size=1, groups=groups)
        self.adapter_mid_1 = nn.Conv2d(8, 8, kernel_size=1, groups=groups)
        nn.init.zeros_(self.adapter_mid_1.bias)
        nn.init.zeros_(self.adapter_mid_1.weight)
        nn.init.zeros_(self.adapter_down_1.weight)
        nn.init.zeros_(self.adapter_down_1.bias)
        nn.init.zeros_(self.adapter_up_1.weight)
        nn.init.zeros_(self.adapter_up_1.bias)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, D1, D2 = x.shape 
        x_down = self.adapter_down_1(x)
        x_down = self.adapter_mid_1(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up_1(x_down).view(B, D1, D2, C)
        return x_up

class Bi_direct_adapter_2(nn.Module):
    def __init__(self, dim=2, input_size=1280):
        super().__init__()
        groups = 4 
        self.adapter_down_2 = nn.Conv2d(input_size, 8, kernel_size=1, groups=groups)
        self.adapter_up_2 = nn.Conv2d(8, input_size, kernel_size=1, groups=groups)
        self.adapter_mid_2 = nn.Conv2d(8, 8, kernel_size=1, groups=groups)
        nn.init.zeros_(self.adapter_mid_2.bias)
        nn.init.zeros_(self.adapter_mid_2.weight)
        nn.init.zeros_(self.adapter_down_2.weight)
        nn.init.zeros_(self.adapter_down_2.bias)
        nn.init.zeros_(self.adapter_up_2.weight)
        nn.init.zeros_(self.adapter_up_2.bias)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, D1, D2 = x.shape 
        x_down = self.adapter_down_2(x)
        x_down = self.adapter_mid_2(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up_2(x_down).view(B, D1, D2, C)
        return x_up

class Bi_direct_adapter_3(nn.Module):
    def __init__(self, dim=2, input_size=2560):
        super().__init__()
        groups = 4  
        self.adapter_down_3 = nn.Conv2d(input_size, 8, kernel_size=1, groups=groups)
        self.adapter_up_3 = nn.Conv2d(8, input_size, kernel_size=1, groups=groups)
        self.adapter_mid_3 = nn.Conv2d(8, 8, kernel_size=1, groups=groups)
        nn.init.zeros_(self.adapter_mid_3.bias)
        nn.init.zeros_(self.adapter_mid_3.weight)
        nn.init.zeros_(self.adapter_down_3.weight)
        nn.init.zeros_(self.adapter_down_3.bias)
        nn.init.zeros_(self.adapter_up_3.weight)
        nn.init.zeros_(self.adapter_up_3.bias)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, D1, D2 = x.shape
        x_down = self.adapter_down_3(x)
        x_down = self.adapter_mid_3(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up_3(x_down).view(B, D1, D2, C)
        return x_up

class CrossAttentionWithForgetGate_0(nn.Module):
    def __init__(self, input_size=320, output_size=64, linear_dim=160):
        super(CrossAttentionWithForgetGate_0, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(in_channels=output_size, out_channels=input_size, kernel_size=1)
        self.linear_dim = linear_dim
        self.output_size = output_size
        self.linear_query = torch.nn.Linear(linear_dim, linear_dim)
        self.linear_key = torch.nn.Linear(linear_dim, linear_dim) 
        self.linear_value = torch.nn.Linear(linear_dim, linear_dim)
        self.conv_forget_attended = nn.Conv1d(output_size, output_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.forget_gate_linear = torch.nn.Linear(linear_dim, linear_dim)
    def forward(self, feature1, feature2):
        B = feature1.shape[0]
        W = feature1.shape[2]
        feature1_conv1 = self.conv1x1(feature1)
        feature2_con1 = self.conv1x1(feature2)
        q_tensor1 = self.linear_query(feature1_conv1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        k_tensor2 = self.linear_key(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        v_tensor2 = self.linear_value(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        attention_scores = torch.matmul(q_tensor1, k_tensor2.transpose(-2, -1)) / (self.linear_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output_tensor = torch.matmul(attention_weights, v_tensor2)
        forget_gate = torch.sigmoid(self.forget_gate_linear(output_tensor.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim))
        output_tensor = self.conv1x1_1(output_tensor * forget_gate + feature2_con1)
        return output_tensor

class CrossAttentionWithForgetGate_1(nn.Module):
    def __init__(self, input_size=640, output_size=128, linear_dim=80):
        super(CrossAttentionWithForgetGate_1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(in_channels=output_size, out_channels=input_size, kernel_size=1)
        self.linear_dim = linear_dim
        self.output_size = output_size
        self.linear_query = torch.nn.Linear(linear_dim, linear_dim) 
        self.linear_key = torch.nn.Linear(linear_dim, linear_dim) 
        self.linear_value = torch.nn.Linear(linear_dim, linear_dim) 
        self.conv_forget_attended = nn.Conv1d(output_size, output_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.forget_gate_linear = torch.nn.Linear(linear_dim, linear_dim)
    def forward(self, feature1, feature2):
        B = feature1.shape[0]
        W = feature1.shape[2]
        feature1_conv1 = self.conv1x1(feature1)
        feature2_con1 = self.conv1x1(feature2)
        q_tensor1 = self.linear_query(feature1_conv1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        k_tensor2 = self.linear_key(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        v_tensor2 = self.linear_value(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        attention_scores = torch.matmul(q_tensor1, k_tensor2.transpose(-2, -1)) / (self.linear_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output_tensor = torch.matmul(attention_weights, v_tensor2)
        forget_gate = torch.sigmoid(self.forget_gate_linear(output_tensor.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim))
        output_tensor = self.conv1x1_1(output_tensor * forget_gate + feature2_con1)
        return output_tensor

class CrossAttentionWithForgetGate_2(nn.Module):
    def __init__(self, input_size=1280, output_size=256, linear_dim=40):
        super(CrossAttentionWithForgetGate_2, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(in_channels=output_size, out_channels=input_size, kernel_size=1)
        self.linear_dim = linear_dim
        self.output_size = output_size
        self.linear_query = torch.nn.Linear(linear_dim, linear_dim)  
        self.linear_key = torch.nn.Linear(linear_dim, linear_dim)  
        self.linear_value = torch.nn.Linear(linear_dim, linear_dim) 
        self.conv_forget_attended = nn.Conv1d(output_size, output_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.forget_gate_linear = torch.nn.Linear(linear_dim, linear_dim)
    def forward(self, feature1, feature2):
        B = feature1.shape[0]
        W = feature1.shape[2]
        feature1_conv1 = self.conv1x1(feature1)
        feature2_con1 = self.conv1x1(feature2)
        q_tensor1 = self.linear_query(feature1_conv1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        k_tensor2 = self.linear_key(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        v_tensor2 = self.linear_value(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim)
        attention_scores = torch.matmul(q_tensor1, k_tensor2.transpose(-2, -1)) / (self.linear_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output_tensor = torch.matmul(attention_weights, v_tensor2)
        forget_gate = torch.sigmoid(self.forget_gate_linear(output_tensor.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim))
        output_tensor = self.conv1x1_1(output_tensor * forget_gate + feature2_con1)
        return output_tensor

class CrossAttentionWithForgetGate_3(nn.Module):
    def __init__(self, input_size=2560, output_size=512, linear_dim=20):
        super(CrossAttentionWithForgetGate_3, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(in_channels=output_size, out_channels=input_size, kernel_size=1)
        self.linear_dim = linear_dim
        self.output_size = output_size
        self.linear_query = torch.nn.Linear(linear_dim, linear_dim) 
        self.linear_key = torch.nn.Linear(linear_dim, linear_dim) 
        self.linear_value = torch.nn.Linear(linear_dim, linear_dim)
        self.conv_forget_attended = nn.Conv1d(output_size, output_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.forget_gate_linear = torch.nn.Linear(linear_dim, linear_dim)
    def forward(self, feature1, feature2):
        B = feature1.shape[0]
        W = feature1.shape[2]
        feature1_conv1 = self.conv1x1(feature1)
        feature2_con1 = self.conv1x1(feature2)
        q_tensor1 = self.linear_query(feature1_conv1.view(-1, self.linear_dim)).view(B, self.output_size, -1, self.linear_dim)
        k_tensor2 = self.linear_key(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, -1, self.linear_dim)
        v_tensor2 = self.linear_value(feature2_con1.view(-1, self.linear_dim)).view(B, self.output_size, -1, self.linear_dim)
        attention_scores = torch.matmul(q_tensor1, k_tensor2.transpose(-2, -1)) / (self.linear_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output_tensor = torch.matmul(attention_weights, v_tensor2)
        forget_gate = torch.sigmoid(self.forget_gate_linear(output_tensor.view(-1, self.linear_dim)).view(B, self.output_size, W, self.linear_dim))
        output_tensor = self.conv1x1_1(output_tensor * forget_gate + feature2_con1)
        return output_tensor

