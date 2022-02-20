from torch import Tensor
import torch
from einops import rearrange, reduce


def conam(self, input: Tensor) -> Tensor:
    global_feature = reduce(input, 'b c h w -> b c', reduction='mean')
    global_feature = rearrange(global_feature, 'b c -> b 1 c ')

    local_feature = rearrange(input, 'b c (h h1) (w w1) -> b c h w h1 w1', h1=self.patch_size, w1=self.patch_size)
    input_ = local_feature
    local_feature = reduce(local_feature, 'b c h w h1 w1 -> b c h w', reduction='mean')
    h = local_feature.shape[2]
    local_feature = rearrange(local_feature, 'b c h w -> b (h w) c')#( b p c)
    mix_local_global = torch.cat([local_feature, global_feature], 1)#(b p+1 c)
    mix_local_global = self.linear1(mix_local_global) 
    mix_local_global = self.relu(mix_local_global)
    mix_local_global = self.linear2(mix_local_global)
    mix_local_global = self.relu(mix_local_global)
    local_feature, global_feature = torch.split(mix_local_global, [local_feature.shape[1], global_feature.shape[1]], 1)#(b p c), (b 1 c)
    global_feature = rearrange(global_feature, 'b p c -> b c p')#(b c 1)
    
    attention = torch.matmul(local_feature, global_feature)#(b p 1)
    attention = reduce(attention, 'b p c -> b p', reduction='mean')# c=1
    attention = self.softmax(attention)
    attention = rearrange(attention, 'b (h w) -> b 1 h w', h=h) # c=1
    attention = rearrange(attention, 'b c h w -> b c h w 1 1')
    input_ = input_ * attention
    input_ = rearrange(input_, 'b c h w h1 w1 -> b c (h h1) (w w1)')
    input = input + input_ #shortcut
    return input

def forward(self, x: Tensor) -> Tensor:
    return self.Attention(x)