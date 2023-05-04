#person relation model 
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import random
from .position_embedding import get_position_embedding

__all__ = ['fbil']

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor+pos

    def forward(self, x, mask=None, pos=None):
        q=k=self.with_pos_embed(x,pos)
        out = self.self_attn(q,k,x,key_padding_mask=mask)[0]
        x = x + self.dropout(out)
        x = self.norm(x)
        return x 

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, q, kv ,mask=None):
        out = self.multihead_attn(q,kv,kv,key_padding_mask=mask)[0]
        q = q + self.dropout(out)
        q = self.norm(q)
        return q

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,x):
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(out)
        x = self.norm(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, d_ffn=2048,dropout=0.1, depth=1):
        super(SelfAttention, self).__init__()
        self.depth=depth
        self.self_atten = nn.ModuleList()
        self.ffn = nn.ModuleList()
        for _ in range(self.depth):
            self.self_atten.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn.append(FFNLayer(d_model, d_ffn, dropout))
        self.pos_embedding = get_position_embedding(d_model=d_model, t='sine')

    def forward(self, src, mask=None):
        n,c,h,w=src.shape
        pos_mask = torch.full((n, h, w),False, device=src.device)
        pos = self.pos_embedding(src, pos_mask)
        l=h*w
        out= src.contiguous().view(n,c,l).transpose(1, 2).transpose(0, 1) # L, N, C
        pos_emb = pos.contiguous().view(n,c,l).transpose(1, 2).transpose(0, 1)    #L, N, C
        for i in range(self.depth):
            out=self.self_atten[i](out, mask=None, pos=pos_emb)
            out=self.ffn[i](out)
        out = out.transpose(0,1).transpose(1,2).view(n, c, h ,w)
        return out


class MaskGuideAttention(nn.Module):
    def __init__(self, d_model, nhead, d_ffn=2048,dropout=0.1, depth=1):
        super(MaskGuideAttention, self).__init__()
        self.depth = depth
        self.nhead = nhead
        self.self_atten = nn.ModuleList()
        self.cross_atten = nn.ModuleList()
        self.ffn = nn.ModuleList()

        for _ in range(self.depth):
            self.self_atten.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.cross_atten.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.ffn.append(FFNLayer(d_model, d_ffn, dropout))

    def forward(self, q, kv, mask=None):
        # q, kv: N, C, H, W
        # mask: N*head, S
        qn,qc,qh,qw = q.shape   
        kn,kc,kh,kw = kv.shape
        l = qh*qw
        s = kh*kw
        out= q.contiguous().view(qn,qc,l).transpose(1, 2).transpose(0, 1) # L, N, C
        kv = kv.contiguous().view(kn,kc,s).transpose(1, 2).transpose(0, 1) # S, N, C
        # if mask is not None:
        #     mask = mask.unsqueeze(1).expand(qn*self.nhead,l,s)
        for i in range(self.depth):
            out = self.self_atten[i](out,mask=None)
            out = self.cross_atten[i](out,kv,mask=mask)
            out = self.ffn[i](out)
        out = out.transpose(0,1).transpose(1,2).view(qn, qc, qh ,qw)
        return out



class FBILHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, act = 'sigmoid', e_depth=1, c_depth=1, l_depth=3, head=8, 
                 local_beta=0.9, local_delta=0.2,global_beta=0.2,global_delta=1.0):
        super(FBILHead, self).__init__()
        self.local_beta = local_beta
        self.local_delta = local_delta
        self.global_beta = global_beta
        self.global_delta = global_delta
        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)
        self.head = head
        # feature encoder
        self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)
        self.feature_encoder = SelfAttention(d_model=reduce_dim, nhead=self.head, d_ffn=2048, dropout=0.1, depth=e_depth)
        # actor global context relation feature
        self.context_atten = MaskGuideAttention(d_model=reduce_dim, nhead=self.head, d_ffn=2048, dropout=0.1, depth=c_depth)
        # actor local context relation feature
        self.local_atten = MaskGuideAttention(d_model=reduce_dim, nhead=self.head, d_ffn=2048, dropout=0.1, depth=l_depth)
        # AFA
        self.weight_linear = MLP(input_dim=reduce_dim,hidden_dim=reduce_dim,output_dim=2,num_layers=3)
        if act == 'sigmoid':
            self.weight_act=nn.Sigmoid()
        elif act == 'softmax':
            self.weight_act=nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(reduce_dim)
        # classification
        self.fc = nn.Linear(reduce_dim*2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    def get_masked(self, n, h, w, rois, beta, delta):
        '''
        For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend.
        '''
        a_rois=rois+0
        a_rois[:, 1] = rois[:, 1] * w
        a_rois[:, 2] = rois[:, 2] * h
        a_rois[:, 3] = rois[:, 3] * w
        a_rois[:, 4] = rois[:, 4] * h
        mask = torch.zeros([n,h,w], device=rois.device).bool()
        # multi actors
        if n > 1:
            # round 
            x1= int(min(a_rois[:,1]))
            y1 = int(min(a_rois[:,2]))
            x2 = math.ceil(max(a_rois[:,3]))
            y2 = math.ceil(max(a_rois[:,4]))
        # one actor
        elif n == 1:
            roi = a_rois[0]
            cx = (roi[1]+roi[3])/2
            cy = (roi[2]+roi[4])/2
            rh = roi[4]-roi[2]
            rw = roi[3]-roi[1]
            x1 = cx-rw if (cx-rw)>0 else 0.0
            y1 = cy-rh if (cy-rh)>0 else 0.0
            x2 = cx+rw if (cx+rw)<w else w
            y2 = cy+rh if (cy+rh)<h else h
            x1=int(x1)
            y1=int(y1)
            x2=math.ceil(x2)
            y2=math.ceil(y2)
        area1 = []
        area2 = []
        for i in range(w):
            for j in range(h):
                if i >= x1 and i <= x2 and j >= y1 and j <= y2:
                    area1.append((i, j))
                else:
                    area2.append((i, j))
        num_area1 = int(len(area1) * beta)
        num_area2 = int(len(area2) * delta)
        for i in range(num_area1):
            idx = random.randint(0, len(area1) - 1)
            x, y = area1[idx]
            mask[:,y,x] = 1
            area1.pop(idx)
        for i in range(num_area2):
            idx = random.randint(0, len(area2) - 1)
            x, y = area2[idx]
            mask[:,y,x] = 1
            area2.pop(idx)
        return mask.view(mask.shape[0], -1)

    def forward(self, data):
        if not isinstance(data['features'], list):
            feats = [data['features']]
        else:
            feats = data['features']
        # temporal average pooling
        h, w = feats[0].shape[3:]
        # requires all features have the same spatial dimensions
        feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
        feats = torch.cat(feats, dim=1)     # torch.Size([3, 2304, 16, 29])
        feats = self.conv_reduce(feats)     # torch.Size([3, 1024, 16, 29])

        rois = data['rois']
        rois_re = rois+0 #deep copy
        rois[:, 1] = rois[:, 1] * w
        rois[:, 2] = rois[:, 2] * h
        rois[:, 3] = rois[:, 3] * w
        rois[:, 4] = rois[:, 4] * h
        rois = rois.detach()   
        roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))    #torch.Size([8, 1024, 7, 7])
        roi_feats = self.roi_maxpool(roi_feats).view(data['num_rois'], -1)      #torch.Size([8, 1024])

        roi_ids = data['roi_ids']
        sizes_before_padding = data['sizes_before_padding']
        high_order_feats = []
        relation_feats = []
        for idx in range(feats.shape[0]):  # iterate over mini-batch
            n_rois = roi_ids[idx+1] - roi_ids[idx]
            if n_rois == 0:
                continue
            actor_feats = roi_feats[roi_ids[idx]:roi_ids[idx+1]].unsqueeze(2).unsqueeze(2)        #torch.Size([3, 1024,1,1])
            actor_guide_feats = actor_feats.squeeze(2).squeeze(2)
            eff_h, eff_w = math.ceil(h * sizes_before_padding[idx][1]), math.ceil(w * sizes_before_padding[idx][0])
            # get mask
            batch_rois = rois_re[roi_ids[idx]:roi_ids[idx+1]]
            bg_feats = feats[idx][:, :eff_h, :eff_w]        #torch.Size([1024, 16, 22])
            bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1))      #torch.Size([3, 1024, 16, 22])
            bg_feats = self.feature_encoder(bg_feats)
            local_mask = self.get_masked(n_rois, eff_h, eff_w, batch_rois, beta=self.local_beta, delta=self.local_delta)
            global_mask = self.get_masked(n_rois, eff_h, eff_w, batch_rois, beta=self.global_beta, delta=self.global_delta)
            # actor local context relation
            l_interact_feats = self.local_atten(q=actor_feats, kv=bg_feats, mask=global_mask)
            # actor global context relation
            g_interact_feats = self.context_atten(q=actor_feats, kv=bg_feats, mask=local_mask)
            l_interact_feats = l_interact_feats.squeeze(2)
            g_interact_feats = g_interact_feats.squeeze(2)
            weight = self.weight_linear(actor_guide_feats)
            # AFA
            weight = self.weight_act(weight).unsqueeze(2)
            relation_feats = torch.cat([l_interact_feats,g_interact_feats], dim=2)
            final_feats = torch.bmm(relation_feats, weight).squeeze(2)
            final_feats = self.norm(final_feats)
            high_order_feats.append(final_feats)
        outputs = torch.cat(high_order_feats, dim=0).view(data['num_rois'], -1)
        outputs = torch.cat([roi_feats, outputs], dim=1)
        if self.dp is not None:
            outputs = self.dp(outputs)
        outputs = self.fc(outputs)
        return {'outputs': outputs}


def fbil(**kwargs):
    model = FBILHead(**kwargs)
    return model
