import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np
import json

class SpatialAttention(nn.Module):
    def __init__(self, d_model, dropout, window_size, linear=False):
        super(SpatialAttention, self).__init__()
        self.embed_dim = d_model
        self.window_size = window_size  # 窗口大小，用于局部稀疏注意力

        self.query = nn.Linear(d_model, d_model) if linear else nn.Identity()
        self.key = nn.Linear(d_model, d_model) if linear else nn.Identity()
        self.value = nn.Linear(d_model, d_model) if linear else nn.Identity()

        self.scale = d_model ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # """
        # 输入 x 的形状为 (seq_len, embed_dim)
        # """
        outputs = []
        for k in range(len(x)):
            i_x = x[k]
            # 计算查询、键和值
            Q = self.query(i_x)  # (batch_size, seq_len, embed_dim)
            K = self.key(i_x)    # (embed_dim, seq_len)
            V = self.value(i_x)  # (embed_dim, seq_len)

            # 使用 unfold 获取 K 和 V 的局部窗口
            # 首先对 K 和 V 进行 padding，以处理边界情况
            # import pdb; pdb.set_trace()
            padding = self.window_size
            K_padded = F.pad(K.transpose(0, 1), (padding, padding), mode='constant', value=0)  # (embed_dim, seq_len + 2*padding)
            V_padded = F.pad(V.transpose(0, 1), (padding, padding), mode='constant', value=0)

            # 使用 unfold 提取局部窗口
            K_windows = K_padded.unfold(dimension=1, size=2 * self.window_size + 1, step=1)  # (embed_dim, seq_len, 2*window_size+1)
            V_windows = V_padded.unfold(dimension=1, size=2 * self.window_size + 1, step=1)  # 同上

            # 转置以便计算
            K_windows = K_windows.permute(1, 2, 0)  # (seq_len, 2*window_size+1, embed_dim)
            V_windows = V_windows.permute(1, 2, 0)  # (seq_len, 2*window_size+1, embed_dim)

            # 计算注意力分数
            attn_scores = torch.bmm(Q.unsqueeze(1), K_windows.transpose(1, 2)).squeeze(1) * self.scale  # (seq_len, 2*window_size+1)

            # 对注意力分数进行 softmax
            attn_weights = F.softmax(attn_scores, dim=-1)  # (seq_len, 2*window_size+1)

            # 使用注意力权重对 V_windows 加权求和
            out = torch.bmm(attn_weights.unsqueeze(1), V_windows).squeeze(1)

            # out = torch.bmm(attn_weights.unsqueeze(1), V_windows).squeeze(1)   # (seq_len, embed_dim)
            out = self.dropout(out)
            out = self.norm(out)

            outputs.append(out)
        return outputs




    
class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs



class CrossAttentionLayer_Pos(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, num_inst, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            inst_queries = queries[i][:num_inst]
            other_queries = queries[i][num_inst:]   
            inst_output, _ = self.attn(inst_queries, k, v, attn_mask=attn_mask)
            if self.fix:
                inst_output = self.dropout(inst_output)
            inst_output = inst_output + inst_queries
            if self.fix:
                inst_output = self.norm(inst_output)     
            output = torch.cat([inst_output, other_queries], dim=0)
            outputs.append(output)
        return outputs



class CrossAttentionLayer_Fix(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(queries)):
            k = v = sources
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs
    

class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN_Res(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, ori_x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i, y in enumerate(x):
            z = self.net(y + ori_x[i])
            z = z + y 
            z = self.norm(z)
            out.append(z)
        return out


@MODELS.register_module()
class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, 
                 num_layers, 
                 num_instance_queries, 
                 num_semantic_queries,
                 num_classes, 
                 in_channels, 
                 d_model, 
                 num_heads, 
                 hidden_dim,
                 dropout, 
                 activation_fn, 
                 iter_pred, 
                 attn_mask, 
                 fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
            
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class QueryDecoder_S3DIS(BaseModule):
    """Query decoder.
    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, 
                 num_layers, 
                 num_instance_queries, 
                 num_semantic_queries,
                 num_classes, 
                 in_channels, 
                 d_model, 
                 num_heads, 
                 hidden_dim,
                 dropout, 
                 activation_fn, 
                 iter_pred, 
                 attn_mask, 
                 fix_attention,
                 objectness_flag, 
                 num_des_prototype,
                 num_img_prototype,
                 des_clip_file,
                 img_clip_file,
                 window_size,
                 **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.SiLU(),)
        
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries > 0:
            self.query = nn.Embedding(num_instance_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.SiLU(),
                nn.Linear(d_model, d_model))
            
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        self.spatial_layers = SpatialAttention(d_model, dropout, window_size)

        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
            
        self.out_norm = nn.LayerNorm(d_model)
        
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model),  nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model),  nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

        self.mask_norm = nn.LayerNorm(num_classes+1)

        self.num_classes =num_classes
        self.num_des_proto = num_des_prototype
        self.num_img_proto = num_img_prototype

        des_prototypes = torch.from_numpy(np.load(des_clip_file)).float()
        self.des_prototypes = nn.Parameter(des_prototypes, requires_grad=False)

        img_prototypes = torch.as_tensor(json.load(open(img_clip_file))['features']).float()
        self.img_prototypes = nn.Parameter(img_prototypes, requires_grad=False)
        
        self.num_all_des = des_prototypes.shape[0]
        self.num_all_img = img_prototypes.shape[0]
        
        self.des_projection = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, d_model))
        
        self.img_projection = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, d_model))

    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            if hasattr(self, 'des_prototypes'):
                embed_prototypes = self.des_projection(self.des_prototypes)
                result_query.append(embed_prototypes)
            if hasattr(self, 'img_prototypes'):
                embed_prototypes = self.img_projection(self.img_prototypes)
                result_query.append(embed_prototypes)
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        sem_attns = []
        for i in range(len(queries)):
            # norm_query = l2_normalize(queries[i])
            norm_query = self.out_norm(queries[i])

            # query_attn_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            inst_query = (norm_query[:-(self.num_all_des+self.num_all_img)])

            des_query = (norm_query[-(self.num_all_des+self.num_all_img):-self.num_all_img] + self.des_projection(self.des_prototypes)).reshape(self.num_classes+1, self.num_des_proto, -1).contiguous()
            
            img_query = (norm_query[-self.num_all_img:] + self.img_projection(self.img_prototypes)).reshape(self.num_classes+1, self.num_img_proto, -1).contiguous()

            mean_sem_query = des_query.mean(dim=1)[:-1] + img_query.mean(dim=1)[:-1]
            # n: h*w, k: num_class, m: num_prototype
            cat_querry = torch.cat([inst_query, mean_sem_query], dim=0)

            # inst_proto_attn = torch.einsum('nd,kmd->nmk', (torch.cat([inst_query, mean_sem_query], dim=0)), sem_query)
            # out_seg = torch.amax(inst_proto_attn, dim=1)
            cls_preds.append(self.out_cls(cat_querry))
            # cls_preds.append(self.mask_norm(out_seg))
            
            pred_score = self.out_score(cat_querry) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)

            inst_pred_mask = torch.einsum('nd,md->nm', inst_query, mask_feats[i])
            des_sem_attn = torch.einsum('knd,md->knm', des_query, mask_feats[i])
            img_sem_attn = torch.einsum('knd,md->knm', img_query, mask_feats[i])

            des_sem_pred = torch.amax(des_sem_attn, dim=1)
            img_sem_pred = torch.amax(img_sem_attn, dim=1)

            sem_pred_mask = torch.sum(torch.stack([des_sem_pred, img_sem_pred]), dim=0)
            # sem_pred_mask = torch.max(des_sem_pred, img_sem_pred)

            
            sem_attns.append(torch.cat([des_sem_attn, img_sem_attn], dim=1))
       
            pred_mask = torch.cat([inst_pred_mask, sem_pred_mask[:-1]], dim=0)

            if self.attn_mask:
                query_attn_mask = torch.cat((inst_pred_mask, rearrange(des_sem_attn, 'n m d -> (n m) d'), rearrange(img_sem_attn, 'n m d -> (n m) d')), dim=0)
                attn_mask = (query_attn_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks, sem_attns

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        mask_feats = self.spatial_layers(mask_feats)
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _, sem_attns = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores,
            sem_attns=sem_attns)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        sem_attns = []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))

        mask_feats = self.spatial_layers(mask_feats)


        cls_pred, pred_score, pred_mask, attn_mask, sem_attn = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        sem_attns.append(sem_attn)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask, sem_attn = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            sem_attns.append(sem_attn)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores, 'sem_attns': sem_attn}
            for cls_pred, scores, masks, sem_attn in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1], sem_attns[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            sem_attns=sem_attns[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)
    

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)





@MODELS.register_module()
class QueryDecoder_ScanNet(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, in_channels, num_instance_classes, num_semantic_classes, objectness_flag, 
                 d_model, dropout, 
                 num_semantic_linears, 
                 num_des_prototype,
                 num_img_prototype,
                 des_clip_file,
                 img_clip_file, 
                 window_size,
                  **kwargs):
        super().__init__(in_channels=in_channels, objectness_flag=objectness_flag, 
            num_classes=num_instance_classes, d_model=d_model,dropout = dropout,window_size=window_size, **kwargs)
        assert num_semantic_linears in [1, 2]


        self.num_semantic_classes = num_semantic_classes
        self.num_instance_classes = num_instance_classes
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.num_des_proto = num_des_prototype
        self.num_img_proto = num_img_prototype

        des_prototypes = torch.from_numpy(np.load(des_clip_file)).float()
        self.des_prototypes = nn.Parameter(des_prototypes, requires_grad=False)

        img_prototypes = torch.as_tensor(json.load(open(img_clip_file))['features']).float()
        self.img_prototypes = nn.Parameter(img_prototypes, requires_grad=False)
        
        self.num_all_des = des_prototypes.shape[0]
        self.num_all_img = img_prototypes.shape[0]
        
        self.des_projection = nn.Sequential(
            nn.Linear(des_prototypes.shape[1], 512), nn.GELU(),
            nn.Linear(512, d_model))

        self.img_projection = nn.Sequential(
            nn.Linear(img_prototypes.shape[1], 512), nn.GELU(),
            nn.Linear(512, d_model))
        
        self.spatial_layers = SpatialAttention(d_model, dropout, window_size)
        self.activation = nn.GELU()
    

    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            if hasattr(self, 'des_prototypes'):
                embed_prototypes = self.des_projection(self.des_prototypes)
                result_query.append(embed_prototypes)
            if hasattr(self, 'img_prototypes'):
                embed_prototypes = self.img_projection(self.img_prototypes)
                result_query.append(embed_prototypes)
            result_queries.append(torch.cat(result_query))
        return result_queries


    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, sem_attns = \
            [], [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])

            inst_query = (norm_query[:-(self.num_all_des+self.num_all_img)])
            des_query = norm_query[-(self.num_all_des+self.num_all_img):-self.num_all_img] + self.des_projection(self.des_prototypes)
            img_query = norm_query[-self.num_all_img:] + self.img_projection(self.img_prototypes)

            cls_preds.append(self.out_cls(inst_query))
            
            if last_flag:

                des_query = rearrange(des_query, '(c n) d -> c n d', c=self.num_semantic_classes+1).contiguous()
                img_query = rearrange(img_query, '(c n) d -> c n d', c=self.num_semantic_classes+1).contiguous()

                des_sem_pred = self.activation(torch.einsum('nd,kmd->nkm', inst_query, des_query))
                img_sem_pred = self.activation(torch.einsum('nd,kmd->nkm', inst_query, img_query))
                
                des_sem_pred_ = (torch.amax(des_sem_pred, dim=-1))
                img_sem_pred_ = (torch.amax(img_sem_pred, dim=-1))

                mix_pred = torch.sum(torch.stack([des_sem_pred_, img_sem_pred_]), dim=0)
                sem_preds.append(mix_pred)
                sem_pred = torch.cat([des_sem_pred, img_sem_pred], dim=-1)
                sem_attns.append(rearrange(sem_pred, 'n k m -> k m n'))

            pred_score = self.out_score(inst_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            attn_mask_ = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            pred_mask = attn_mask_[:-(self.num_all_des+self.num_all_img)]

            if self.attn_mask:
                attn_mask = (attn_mask_.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, sem_attns

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]

        spatial_feat = self.spatial_layers(mask_feats)
        mask_feats = [0.1 * sf +  mf for sf, mf in zip(spatial_feat, mask_feats)]

        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _, sem_attns = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores,
            sem_attns=sem_attns)
    

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks, sem_attns = [], [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]

        spatial_feat = self.spatial_layers(mask_feats)
        mask_feats = [0.1 * sf +  mf for sf, mf in zip(spatial_feat, mask_feats)]

        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask, sem_attn = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        sem_attns.append(sem_attn)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask, sem_attn = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            sem_attns.append(sem_attn)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores,
                sem_attns=sem_attn)
            for cls_pred, sem_pred, scores, masks, sem_attn in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1], sem_attns[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            sem_attns=sem_attns[-1],
            aux_outputs=aux_outputs)