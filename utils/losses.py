import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal, Beta
import copy
from sklearn.preprocessing import MinMaxScaler

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss #.mean()

class FocalLoss(nn.Module):
    def __init__(self, config, weight=None, gamma=2.0):
        super().__init__()
        self.config = config
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, config, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        self.config = config
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, target):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, config, cls_num_list, beta=0.9999):
        super().__init__()
        self.config = config
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class GeneralizedReweightLoss(nn.Module):
    def __init__(self, config, cls_num_list, exp_scale=1.0):
        super().__init__()
        self.config = config
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, config, cls_num_list):
        super().__init__()
        self.config = config
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, target):
        logit_adjusted = logit + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, config, cls_num_list, tau=1.0):
        super().__init__()
        self.config = config
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.cls_num_list = cls_num_list
        self.tau = tau
        freq = torch.log(cls_num_list)
        self.freq = tau*(freq-freq.min())/(freq.max()-freq.min()+1e-5)
               
    def forward(self, logit, target, reduction='mean'):
        #tau = self.freq[target]
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target, reduction=reduction)
             

class TopKLogitAdjustedLoss(nn.Module):
    def __init__(self, config, cls_num_list, tau=1., min_k=30, max_k=100):
        super().__init__()
        self.config = config
        self.cls_num_list = cls_num_list
        cls_freq = cls_num_list/cls_num_list.max()
        cls_inv_freq = cls_num_list.max()/cls_num_list
        self.cls_inv_freq = cls_inv_freq/cls_inv_freq.max()
        cls_num_ratio = cls_num_list / cls_num_list.sum()
        self.log_cls_num = torch.log(cls_num_ratio + 1e-6)
        self.tau = tau
       
        # K 与类别样本数量成反比（常见类对应更小的 K）
        freq = (cls_num_list.max() / cls_num_list).float() #self.log_cls_num.max() / self.log_cls_num #cls_num_list.float() #
        freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-6)        
        k_per_class = (freq_norm * (max_k - min_k) + min_k).long()  # [C]
        self.register_buffer("k_per_class", k_per_class)

    def forward(self, logit, target, reduction='mean'):
        """
        基于 logit adjustment 和 top-k soft target 的融合版本：
        - 非目标类加 logit 先验提升尾类
        - top-k 类用 soft target 辅助监督
        """
        device = logit.device
        B, C = logit.size()

        logit_adjusted = (logit + self.tau * self.log_cls_num.unsqueeze(0))
        #logit_adjusted2 = logit + tau * self.log_cls_num.unsqueeze(0)

        # standard LA loss
        loss_full = F.cross_entropy(logit_adjusted, target, reduction=reduction)

        # === Step 2: softmax with top-K probability masking ===
        prob_adjusted = F.softmax(logit_adjusted, dim=1)  # [B, C]
        masked_prob = torch.zeros_like(prob_adjusted)
        k_list = self.k_per_class[target]  # 每个样本用不同的 K 值
        
        for i in range(B):
            k = min(k_list[i].item(), C)
            topk_val, topk_idx = torch.topk(logit[i], k=k)
            #if target[i] not in topk_idx:
            #    topk_idx[-1] = target[i]  # 强制替换最后一个为 target 类           
            masked_prob[i, topk_idx] = prob_adjusted[i, topk_idx]

        masked_prob = masked_prob + 1e-6
        masked_prob = masked_prob / masked_prob.sum(dim=1, keepdim=True)  # 重新归一化
        log_prob = torch.log(masked_prob)
        loss_topk = F.nll_loss(log_prob, target, reduction=reduction)
        
    
        # === Step 3: Combine ===
        loss = 0.5*loss_full + 0.5*loss_topk

        return loss #.mean()


class DBMargin(nn.Module):
    def __init__(self, config, cls_num_list, tau=1.0, max_margin = 0.1, lambda_inst = 1.0): #, self, args, dist
        super(DBMargin, self).__init__()       
        self.n_classes = len(cls_num_list) 
        self.scale = config.scale       
        
        self.dist = cls_num_list #torch.from_numpy(np.array(dist)).float().cuda()
        self.prob = self.dist / sum(self.dist)
        self.margin = ((1 / self.prob ** tau) / (1 / self.prob ** tau)[-1]).unsqueeze(0) * max_margin
        self.lambda_inst = lambda_inst


        self.config = config
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.cls_num_list = cls_num_list
        self.tau = tau
            
    def forward(self, logits, target):    
        target_oh = F.one_hot(target, num_classes=self.n_classes)
        incorrect = logits.argmax(dim=-1) != target
        margin_seed = self.margin * target_oh
        margin_cls = margin_seed
        
        cosine = logits/self.scale
        inst_diff = (1 - cosine) / 2 
        margin_inst = margin_cls * inst_diff * incorrect.unsqueeze(-1)
        margin = margin_cls + self.lambda_inst * margin_inst 
                   
        logits = self.scale * torch.cos(torch.acos(cosine) + margin) 
        
        
        logit_adjusted = logits + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)  


class LADELoss(nn.Module):
    def __init__(self, config, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.config = config
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, target):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, target)

        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss
    
    
class GCLLoss(nn.Module):
    def __init__(self, config, cls_num_list, m=0.1, train_cls=False, noise_mul=1., gamma=0.):
        super(GCLLoss, self).__init__()
        self.config = config
        cls_list = cls_num_list.clone().to('cuda') #torch.tensor(cls_num_list, dtype=torch.float, device='cuda') #torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list/m_list.max()
        self.m = m
        self.simpler = normal.Normal(0, 1 / 3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
        self.reweight_epoch  = config.reweight_epoch
        if self.reweight_epoch != -1:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list.cpu())
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).to('cuda')
        else:
            self.per_cls_weights_enabled = None
            self.per_cls_weights = None
        if self.config.EMO_enable==True:
            self.smooth_label = SoftLabelGenerator(len(cls_list), self.m_list)

    def _hook_before_epoch(self, epoch, cost_embedding):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
                self.cost_embedding = copy.deepcopy(cost_embedding)
            else:
                self.per_cls_weights = None   
                self.cost_embedding = None 

    def _update_cost_embed(self, cost_embedding):
        self.cost_embedding = copy.deepcopy(cost_embedding)

    def forward(self, logits, targets):
        # ======================================================================== #
        #                   Compute the MLE loss
        # ======================================================================== #
        
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)

        noise = self.simpler.sample(logits.shape).clamp(-1, 1).to(logits.device)  # self.scale(torch.randn(cosine.shape).to(cosine.device))

        # cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list
        logits = logits - self.config.scale*self.noise_mul * noise.abs() * self.m_list
        output = torch.where(index.bool(), logits - self.config.scale*self.m, logits)
        if self.train_cls:
            mle_loss = focal_loss(F.cross_entropy(output, targets, reduction='none', 
                                                  weight=self.per_cls_weights), self.gamma)
        else:
            mle_loss = F.cross_entropy(output, targets, 
                                       weight=self.per_cls_weights,reduction='none')
        
        if (self.config.EMO_enable==True) and (self.epoch>self.reweight_epoch):
            # ======================================================================== #
            #                   Compute the EMO loss
            # ======================================================================== #       
            labels_tmp = targets.clone()
            one_hot = torch.nn.functional.one_hot(labels_tmp, num_classes=len(self.m_list)).to(logits.dtype)
            #stable_onehot = (one_hot+1e-9) / torch.linalg.vector_norm((one_hot+1e-15), ord=1, dim=-1, keepdim=True) # (bsz*seq_len, vocab_size)
            soft_label = self.smooth_label(one_hot)
            embedding_matrix = self.cost_embedding # (class_size, hidden_size)
            embedding_matrix = embedding_matrix / (1e-15+ torch.linalg.vector_norm(embedding_matrix, ord=2, dim=1, keepdim=True))
            p_contextual_repr = soft_label @ embedding_matrix #stable_onehot @ embedding_matrix # (bsz, hidden_size), weights of target class, EP in Eq.14  
            q_grad = torch.log_softmax(logits, dim=-1).exp() # (bsz, class_size), logit-->softmax probability
            gt_q = (q_grad * one_hot).detach()
            q_final = q_grad - gt_q  # remove the softmax probability of target class
            q_contextual_repr = q_final @ embedding_matrix # (bsz, hidden_size) 按概率组合类中心? E{Q_\theta} in Eq.14
            emo_loss = (1 - torch.sum(p_contextual_repr*q_contextual_repr, dim=-1)) # (bsz*seq_len,)
            loss = ((mle_loss / (emo_loss+1e-15)).detach() * emo_loss + mle_loss ) * 0.5
        else: 
            loss = mle_loss

        return loss.mean()  




    
# class EMOLoss(nn.Module):
#     def __init__(self, config, cls_num_list=None, cost_embedding = None):
#         super().__init__()
#         self.config = config
#         self.cost_embedding = cost_embedding
#         cls_list = torch.cuda.FloatTensor(cls_num_list)
#         m_list = torch.log(cls_list)
#         m_list = m_list- m_list.min() #m_list.max() - m_list #
#         self.m_list = m_list /m_list.max()* config.loss_type_EMO_m
#         self.smooth_label = SoftLabelGenerator(len(cls_list), self.m_list)
#         self.eopch = 0

#     def _hook_before_epoch(self, epoch):       
#         #if self.reweight_epoch != -1:
#         self.epoch = epoch

#         #    if epoch > self.reweight_epoch:
#         #        self.per_cls_weights = self.per_cls_weights_enabled
#         #    else:
#         #        self.per_cls_weights = None        

#     def _update_cost_embed(self, cost_embedding):
#         self.cost_embedding = copy.deepcopy(cost_embedding)

        
#     def forward(self, logits, targets):
#         # ======================================================================== #
#         #                   Compute the MLE loss
#         # ======================================================================== #
#         mle_loss = F.cross_entropy(logits, targets, reduction='none') #loss_fct(logits, targets)
        
#         if self.epoch>self.config.start_EMO:
#         # ======================================================================== #
#         #                   Compute the EMO loss
#         # ======================================================================== #
#             labels_tmp = targets.clone()
#             one_hot = torch.nn.functional.one_hot(labels_tmp, num_classes=len(self.m_list)).to(logits.dtype)
#             #stable_onehot = (one_hot+1e-9) / torch.linalg.vector_norm((one_hot+1e-15), ord=1, dim=-1, keepdim=True) # (bsz*seq_len, vocab_size)
#             soft_label = self.smooth_label(one_hot)
#             embedding_matrix = self.cost_embedding # (class_size, hidden_size)
#             embedding_matrix = embedding_matrix / (1e-15+ torch.linalg.vector_norm(embedding_matrix, ord=2, dim=1, keepdim=True))
#             p_contextual_repr = soft_label @ embedding_matrix #stable_onehot @ embedding_matrix # (bsz, hidden_size), weights of target class, EP in Eq.14  
#             q_grad = torch.log_softmax(logits, dim=-1).exp() # (bsz, class_size), logit-->softmax probability
#             gt_q = (q_grad * one_hot).detach()
#             q_final = q_grad - gt_q  # remove the softmax probability of target class
#             q_contextual_repr = q_final @ embedding_matrix # (bsz, hidden_size) 按概率组合类中心? E{Q_\theta} in Eq.14
#             emo_loss = (1 - torch.sum(p_contextual_repr*q_contextual_repr, dim=-1)) # (bsz*seq_len,)
#             loss = ((mle_loss / (emo_loss+1e-15)).detach() * emo_loss + mle_loss ) * 0.5
#         else: 
#             loss = mle_loss

#         # ======================================================================== #
#         #                   Compose the final loss
#         # ======================================================================== #     
#         #loss = ((mle_loss / (emo_loss+1e-9)).detach() * emo_loss + mle_loss ) * 0.5
#         #loss = (emo_loss / (mle_loss+1e-10)).detach() * mle_loss + emo_loss

#         return loss.mean()  
        
        
class SoftLabelGenerator:
    def __init__(self, num_classes, class_eps=None):
        """
        初始化soft label生成器
        Args:
            num_classes: 类别总数
            class_eps: 每个类别的平滑系数，可以传入：
                      - None：使用默认值0.1
                      - float：所有类别使用相同平滑系数
                      - list/tensor：为每个类别指定不同平滑系数
        """
        self.num_classes = num_classes
        
        # 初始化平滑系数表
        if class_eps is None:
            self.class_eps = torch.full((num_classes,), 0.1)
        elif isinstance(class_eps, (float, int)):
            self.class_eps = torch.full((num_classes,), float(class_eps))
        else:
            assert len(class_eps) == num_classes, "平滑系数数量必须等于类别数"
            self.class_eps = class_eps 
    
    def __call__(self, one_hot):
        """
        根据one-hot标签生成soft label
        Args:
            one_hot: [N, C]的one-hot标签
        Returns:
            soft_label: 平滑后的标签
        """
        device = one_hot.device
        self.class_eps = self.class_eps.to(device)
        
        # 向量化实现
        mask = one_hot.bool()  # [N, C]
        eps_expanded = self.class_eps.unsqueeze(0)  # [1, C]
        remaining_prob = eps_expanded / (self.num_classes - 1)  # [1, C]
        
        soft_label = torch.zeros_like(one_hot) + remaining_prob  # 初始化为剩余概率
        target_probs = 1 - self.class_eps[one_hot.argmax(dim=1)]
        soft_label = soft_label.clone()  # 确保不是扩展张量
        soft_label[mask] = target_probs
        
        return soft_label


class EMOLoss(nn.Module):
    def __init__(self, config, cls_num_list=None, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        self.cls_num_list = cls_num_list.clone().to(self.device)   
        #class_eps = torch.log(self.cls_num_list)
        #class_eps = (class_eps - class_eps.min())/(class_eps.max()-class_eps.min()) * 0.8
        #self.smooth_label = SoftLabelGenerator(len(cls_num_list), class_eps)       
        #self.class_w = self._scaled_weights(cls_num_list) #m_list-m_list.min() #
        #self.register_buffer('W', m_list/m_list.max())
        #self.mle_loss = LogitAdjustedLoss(config, cls_num_list=cls_num_list) #nn.KLDivLoss(reduction='none')
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        self.log_cls_num = torch.log(cls_num_ratio+ 1e-15)

        #self.reweight_epoch = config.reweight_epoch        
        
    def _scaled_weights(self, cls_num_list, alpha=0.5, norm_range=[0.5,1.5]):
        m_list = torch.log(cls_num_list) #cls_num_list #
        class_w = m_list.max()/m_list
        # 非对称频次惩罚项 w(i,j)
        Ni = class_w.view(-1, 1)  # [C, 1]
        Nj = class_w.view(1, -1)  # [1, C]
        asym_factor = (Nj / (Ni + 1e-15)) ** alpha  # [C, C]        
        # Min-max normalize to [min_val, max_val]
        #asym_factor_scaled = asym_factor/asym_factor.max()
        #min_raw = torch.min(asym_factor)
        #max_raw = torch.max(asym_factor)        
        #asym_factor_norm = (asym_factor - min_raw) / (max_raw - min_raw + 1e-15)        
        #asym_factor_scaled = asym_factor_norm * (norm_range[1] - norm_range[0]) + norm_range[0]  # scale to [0.5, 1.5]                
        return asym_factor #asym_factor_scaled
        
    def _update_cost_embed(self, cost_embedding):
        self.class_embeddings = copy.deepcopy(cost_embedding)

        
    def _build_cost_matrix(self):
        # 语义相似性代价
        embeddings_norm = F.normalize(self.class_embeddings, dim=1)
        semantic_cost = 1 - torch.mm(embeddings_norm, embeddings_norm.T)
            
        # 长尾权重修正融合
        cost_matrix = semantic_cost.fill_diagonal_(0)  # * self.class_w #self.class_w #.fill_diagonal_(0)
        
        #基于类别频次的动态调整对角线元素
        #freq_inv = 1.0 / (torch.log(self.cls_num_list)+ 1e-6) 
        #freq_inv = freq_inv/freq_inv.max()
        #diag_values =  0.005 *  freq_inv  
        #cost_matrix[torch.arange(cost_matrix.size(0)), torch.arange(cost_matrix.size(1))] = diag_values.to(cost_matrix.dtype)
        return cost_matrix 

    def _hook_before_epoch(self, epoch):
        self.epoch = epoch    

    def forward(self, logits, targets):
        labels_tmp = targets.clone()
        one_hot = torch.nn.functional.one_hot(labels_tmp, num_classes=len(self.cls_num_list)).to(logits.dtype)
        #stable_onehot = (one_hot+1e-9) / torch.linalg.vector_norm((one_hot+1e-15), ord=1, dim=-1, keepdim=True) # (bsz*seq_len, vocab_size)
        #soft_label = self.smooth_label(one_hot)
        
        # ======================================================================== #
        #                   Compute the EMO loss
        # ======================================================================== #
        #cost_matrix = self._build_cost_matrix()
        #emo_loss = torch.einsum('bi,ij,bj->b', logits.softmax(dim=-1), cost_matrix, one_hot)
        tau = (self.epoch/self.config.num_epochs)**2 #(1-logits/self.config.scale) #torch.exp(emo_loss.detach()+ 1e-15)/2.
        # ======================================================================== #
        #                   Compute the MLE loss
        # ======================================================================== #
        logit_adjusted = logits + tau * self.log_cls_num.unsqueeze(0).tile(logits.shape[0], 1)
        #logit_adjusted = logits + tau * self.log_cls_num.unsqueeze(0) 
        mle_loss = F.cross_entropy(logit_adjusted, targets) # self.mle_loss(logits, targets, reduction='none') 
                   # loss_fct(logits, targets) self.kl_loss(log_probs, soft_label).mean(1) #        

        # ======================================================================== #
        #                   Compose the final loss
        # ======================================================================== #         
        loss = mle_loss
        #loss = ((mle_loss / (emo_loss+1e-15)).detach() * emo_loss + mle_loss) * 0.5
        #loss = (emo_loss / (mle_loss+1e-15)).detach() * mle_loss + emo_loss

        return loss    
    
    