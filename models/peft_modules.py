import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
    
    @property
    def dtype(self):
        return self.prompt.dtype

    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    @property
    def dtype(self):
        return self.ln.weight.dtype
    
    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x ):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x


'''
class ClassAwareDropout(nn.Module):
    def __init__(self, dropout_p: torch.Tensor):
        super().__init__()
        self.register_buffer("dropout_p", dropout_p)  # [num_classes]

    def forward(self, x, target):
        # x: [B, ...]
        B = x.size(0)
        keep_prob = 1.0 - self.dropout_p[target].view(1, -1, 1)  # broadcast to x shape
        mask = (torch.rand_like(x) < keep_prob).float()
        return x * mask / (keep_prob + 1e-6)   # scale to keep expected value
'''

class ClassWiseAdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, cls_num_list= None, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))
        
        # === Class Frequency ===
        self.cls_num_list = cls_num_list
        self.cls_freq = cls_num_list/cls_num_list.sum()
        cls_inv_freq = self.cls_num_list.max()/self.cls_num_list
        self.cls_inv_freq = cls_inv_freq/cls_inv_freq.sum()
        
        
        # FiLM Class Embedding & Feature Modulation
        num_classes = len(cls_num_list)
        self.class_embed = nn.Embedding(num_classes, bottle_dim)
        #self.mod_proj = nn.Linear(bottle_dim, bottle_dim)
        self.mod_proj = nn.Sequential(
            nn.Linear(bottle_dim + bottle_dim, bottle_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottle_dim, bottle_dim)
            )
        
        # Initialization
        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        #for film 
        nn.init.normal_(self.class_embed.weight, std=0.02)
        # 手动初始化 mod_proj 的 Linear 层
        for m in self.mod_proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)
        # For reference (no label), average γ / β 
        self.register_buffer("avg_class_embed", torch.zeros(bottle_dim))
        self.register_buffer("seen_count", torch.tensor(0.))

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x, target=None, fallback=False):
        L, B, D = x.shape
        x = self.ln(x)
        x = self.down_proj(x)        
        
        if target is not None and not fallback:          
            class_embed = self.class_embed(target) # class_embed: [B, bottle_dim]
            class_embed = class_embed.unsqueeze(0).expand(L, -1, -1)  # [L, B, bottle_dim]
            
            # update running average
            with torch.no_grad():
                # class_embed running average
                #self.avg_class_embed = (self.avg_class_embed * self.seen_count + class_embed.mean(dim=(0, 1))
                #                        ) / (self.seen_count + 1)
                #self.seen_count += 1   
                weights = self.cls_inv_freq[target].view(B, 1)  # [B, 1]
                weighted_embed = class_embed * weights  
                total_weight = weights.sum()
                mean_embed = weighted_embed.sum(dim=(0, 1)) / (total_weight + 1e-6) 
                # 更新 running average
                self.avg_class_embed = (self.avg_class_embed * self.seen_count + mean_embed) / (self.seen_count + total_weight)
                self.seen_count += total_weight                
                               
        else:
            class_embed = self.avg_class_embed.view(1, 1, -1).expand(L, B, -1)  # [L, B, bottle_dim]

        # concat: [L, B, 2*bottle_dim]
        mod_input = torch.cat([x, class_embed], dim=-1) # [L, B, 2*bottle_dim]
        mod = self.mod_proj(mod_input)  # [L, B, bottle_dim]
        # modulate
        x = x + mod  # residual modulation
        
        x = self.relu(x)                       
        x = self.up_proj(x)
        return x * self.scale
    
    
    

class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim, dtype=dtype))
        self.scaling = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            return x * self.scale + self.shift

class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, ratio=0.0, generator=None):
        super().__init__()
        # weight: (out_dim, in_dim)
        # bias: (out_dim)
        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        ratio = float(eval(ratio)) if isinstance(ratio, str) else float(ratio)
        num_masked = int(num_params * ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[:num_masked]
        mask = torch.zeros(num_params, dtype=bool).scatter(dim=0, index=masked_indexs, value=True)
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:,:-1]
        self.mask_bias = mask[:,-1]

        self.optimized_weight = nn.Parameter(torch.masked_select(weight.detach(), mask=self.mask_weight))
        self.optimized_bias = nn.Parameter(torch.masked_select(bias.detach(), mask=self.mask_bias))

    def forward(self, x, weight, bias):
        self.mask_weight = self.mask_weight.to(weight.device)
        self.mask_bias = self.mask_bias.to(bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(weight, mask=self.mask_weight, source=self.optimized_weight)
        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(bias, mask=self.mask_bias, source=self.optimized_bias)
        return F.linear(x, weight, bias)