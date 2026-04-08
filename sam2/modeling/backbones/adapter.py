import math
import torch
import torch.nn as nn


class Bottleneck_Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",  # 默认采用 lora 初始化
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()

        # 直接写死默认参数，如果 d_model 或 bottleneck 为 None，则使用固定数值
        self.n_embd = 768 if d_model is None else d_model
        self.down_size = 64 if bottleneck is None else bottleneck

        # 设置适配器的层归一化选项
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # 如果选择 learnable_scalar，则设置一个可学习的缩放因子，否则直接转为 float
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # 定义降维线性层和升维线性层
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()  # 使用 GELU 激活函数
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.non_linear_func = nn.GELU()  # 再次使用 GELU

        # 设置丢弃率
        self.dropout = dropout

        # 如果初始化选项为 "bert"，则直接报错；这里我们默认使用 lora 初始化
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            # 使用 He 正态初始化方法初始化降维层，并将升维层的权重和偏置置零
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        up = self.non_linear_func(up)
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        output = up + residual if add_residual else up
        return output