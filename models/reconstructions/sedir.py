import copy
import math
import random
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _labels_from_input(input_dict, device):
    if "cls_label" in input_dict:
        labels = input_dict["cls_label"]
        if torch.is_tensor(labels):
            return labels.long().to(device)
        return torch.tensor(labels, dtype=torch.long, device=device)
    if "category" in input_dict:
        labels = input_dict["category"]
        if torch.is_tensor(labels):
            return labels.long().to(device)
        return torch.tensor(labels, dtype=torch.long, device=device)
    if "label" in input_dict:
        labels = input_dict["label"]
        if torch.is_tensor(labels):
            return labels.long().to(device)
        return torch.tensor(labels, dtype=torch.long, device=device)
    if "clsname" in input_dict:
        labels = []
        for item in input_dict["clsname"]:
            text = str(item)
            digits = "".join(ch for ch in text if ch.isdigit())
            labels.append(int(digits) if digits else abs(hash(text)) % 10000)
        return torch.tensor(labels, dtype=torch.long, device=device)
    return None


def geometry_descriptor(center, eps=1e-6):
    """Return normal/curvature-inspired local variation descriptors per group."""
    if center.size(1) < 3:
        return center.new_zeros(center.size(0), center.size(1), 2)

    diffs = center[:, :, None, :] - center[:, None, :, :]
    dist = torch.linalg.norm(diffs, dim=-1)
    k = min(16, center.size(1))
    _, idx = torch.topk(dist, k=k, largest=False, dim=-1)
    neighbors = torch.gather(
        center[:, None, :, :].expand(-1, center.size(1), -1, -1),
        2,
        idx[..., None].expand(-1, -1, -1, 3),
    )
    local = neighbors - neighbors.mean(dim=2, keepdim=True)
    cov = local.transpose(-1, -2).matmul(local) / max(k - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eps)
    normals = eigvecs[..., 0]
    curvature = eigvals[..., 0] / eigvals.sum(dim=-1).clamp_min(eps)

    neighbor_normals = torch.gather(
        normals[:, None, :, :].expand(-1, center.size(1), -1, -1),
        2,
        idx[..., None].expand(-1, -1, -1, 3),
    )
    neighbor_curvature = torch.gather(
        curvature[:, None, :].expand(-1, center.size(1), -1),
        2,
        idx,
    )
    normal_var = 1.0 - (neighbor_normals * normals[:, :, None, :]).sum(dim=-1).abs()
    normal_var = normal_var.mean(dim=-1)
    curvature_var = (neighbor_curvature - curvature[:, :, None]).abs().mean(dim=-1)
    desc = torch.stack([normal_var, curvature_var], dim=-1)
    mean = desc.mean(dim=1, keepdim=True)
    std = desc.std(dim=1, keepdim=True).clamp_min(eps)
    return (desc - mean) / std


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))

    def forward_pre(self, src, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_key_padding_mask, pos)
        return self.forward_post(src, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class CoarseToFineGlobalTokenizer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        global_dim,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        activation,
        normalize_before,
    ):
        super().__init__()
        layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(layer, num_layers, norm)
        self.act_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.scale_proj = nn.ModuleDict(
            {
                "fine": nn.Linear(hidden_dim, hidden_dim),
                "base": nn.Identity(),
                "coarse": nn.Linear(hidden_dim, hidden_dim),
            }
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 4, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
        )
        nn.init.trunc_normal_(self.act_token, std=0.02)

    def _encode_scale(self, tokens, pos, scale_name):
        projected = self.scale_proj[scale_name](tokens)
        return self.encoder(projected, pos=pos)

    def forward(self, feature_tokens, pos_embed):
        base = self._encode_scale(feature_tokens, pos_embed, "base")
        fine = self._encode_scale(feature_tokens, pos_embed, "fine")
        coarse = self._encode_scale(feature_tokens, pos_embed, "coarse")

        batch = feature_tokens.size(1)
        act = self.act_token.expand(-1, batch, -1)
        act_seq = torch.cat([act, feature_tokens], dim=0)
        zero_pos = pos_embed.new_zeros(1, batch, pos_embed.size(-1))
        act_encoded = self.encoder(act_seq, pos=torch.cat([zero_pos, pos_embed], dim=0))
        act_token = act_encoded[0]

        global_feature = torch.cat(
            [
                base.mean(dim=0),
                coarse.mean(dim=0),
                fine.mean(dim=0),
                act_token,
            ],
            dim=-1,
        )
        global_token = F.normalize(self.projector(global_feature), dim=-1)
        loss_cos = (
            1.0 - F.cosine_similarity(base, fine, dim=-1)
        ).mean() + (1.0 - F.cosine_similarity(base, coarse, dim=-1)).mean()
        return base, fine, coarse, global_feature, global_token, loss_cos


class CategoryContrastiveBuffer(nn.Module):
    def __init__(self, feature_dim, buffer_size, temperature):
        super().__init__()
        self.buffer_size = buffer_size
        self.temperature = temperature
        self.register_buffer("features", torch.zeros(buffer_size, feature_dim))
        self.register_buffer("labels", torch.full((buffer_size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, features, labels):
        features = features.detach()
        labels = labels.detach()
        for feature, label in zip(features, labels):
            idx = int(self.ptr.item() % self.buffer_size)
            self.features[idx].copy_(feature)
            self.labels[idx] = label
            self.ptr[0] += 1

    def loss(self, features, labels):
        valid = self.labels >= 0
        if valid.sum() == 0:
            return features.sum() * 0.0
        bank_features = torch.cat([self.features[valid].detach(), features], dim=0)
        bank_labels = torch.cat([self.labels[valid].detach(), labels], dim=0)
        losses = []
        logits = features.matmul(bank_features.t()) / self.temperature
        for row, label in enumerate(labels):
            positive = bank_labels == label
            positive[valid.sum() + row] = False
            if positive.any():
                log_prob = logits[row] - torch.logsumexp(logits[row], dim=0)
                losses.append(-log_prob[positive].mean())
        if not losses:
            return features.sum() * 0.0
        return torch.stack(losses).mean()


class GeometryGuidedAttention(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout=0.1):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.geo_mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, nhead))
        self.geo_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value, geo_desc):
        batch, groups, _ = key.shape
        q = self.q_proj(query).view(batch, 1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, groups, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, groups, self.nhead, self.head_dim).transpose(1, 2)
        logits = q.matmul(k.transpose(-2, -1)) * self.scale
        geo_bias = self.geo_mlp(geo_desc).permute(0, 2, 1).unsqueeze(2)
        logits = logits + self.geo_strength * geo_bias
        attn = self.dropout(torch.softmax(logits, dim=-1))
        context = attn.matmul(v).transpose(1, 2).reshape(batch, 1, self.hidden_dim)
        return self.out_proj(context).expand(-1, groups, -1)


class GeometryGuidedDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.geo_attn = GeometryGuidedAttention(hidden_dim, nhead, dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tokens, memory, global_token, pos, geo_desc):
        q = k = tokens + pos
        tokens2 = self.self_attn(q, k, value=tokens)[0]
        tokens = self.norm1(tokens + self.dropout1(tokens2))

        guided = self.geo_attn(global_token, (memory + pos).transpose(0, 1), memory.transpose(0, 1), geo_desc)
        tokens = self.norm2(tokens + self.dropout2(guided.transpose(0, 1)))

        tokens2 = self.linear2(self.dropout(self.activation(self.linear1(tokens))))
        return self.norm3(tokens + self.dropout3(tokens2))


class GeometryGuidedDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        activation,
        normalize_before,
    ):
        super().__init__()
        layer = GeometryGuidedDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.layers = _get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, memory, global_token, pos, geo_desc):
        output = memory
        for layer in self.layers:
            output = layer(output, memory, global_token, pos, geo_desc)
        return self.norm(output)


class SeDiR(nn.Module):
    def __init__(
        self,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        initializer,
        cls_num,
        inplanes=384,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        global_dim=None,
        c3l_buffer_size=64,
        contrast_temperature=0.2,
        lambda_scl=0.001,
        lambda_cls=0.001,
        lambda_cos=0.01,
        **kwargs,
    ):
        super().__init__()
        del neighbor_mask, kwargs
        self.feature_size = feature_size
        self.feature_jitter = feature_jitter
        self.cls_num = cls_num
        self.lambda_scl = lambda_scl
        self.lambda_cls = lambda_cls
        self.lambda_cos = lambda_cos
        global_dim = global_dim or hidden_dim

        self.input_proj = nn.Linear(inplanes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes)
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, hidden_dim))
        self.cfgt = CoarseToFineGlobalTokenizer(
            hidden_dim,
            global_dim,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        self.global_to_hidden = nn.Linear(global_dim, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim * 4, cls_num)
        self.c3l = CategoryContrastiveBuffer(global_dim, c3l_buffer_size, contrast_temperature)
        self.decoder = GeometryGuidedDecoder(
            hidden_dim,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        self.geo_desc_proj = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            jitter = torch.randn_like(feature_tokens) * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        feature_align = input["xyz_features"]
        center = input["center"].to(feature_align.device)
        feature_tokens = rearrange(feature_align, "b n g -> g b n")
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        feature_tokens = self.input_proj(feature_tokens)
        pos_embed = self.pos_embed(center).permute(1, 0, 2)

        base, _fine, _coarse, global_feature, global_token, loss_cos_raw = self.cfgt(
            feature_tokens, pos_embed
        )
        labels = _labels_from_input(input, feature_align.device)
        if labels is not None:
            labels = labels.remainder(self.cls_num)
            cls_pred = self.cls_head(global_feature)
            loss_cls_raw = F.cross_entropy(cls_pred, labels)
            loss_scl_raw = self.c3l.loss(global_token, labels)
            if self.training:
                self.c3l.enqueue(global_token, labels)
        else:
            cls_pred = self.cls_head(global_feature)
            zero = global_token.sum() * 0.0
            loss_cls_raw = zero
            loss_scl_raw = zero

        geo_desc = self.geo_desc_proj(geometry_descriptor(center))
        hidden_global = self.global_to_hidden(global_token)
        decoded = self.decoder(base, hidden_global, pos_embed, geo_desc)
        feature_rec_tokens = self.output_proj(decoded)
        feature_rec = rearrange(feature_rec_tokens, "g b n -> b n g")
        pred = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True) + 1e-12)
        loss_rec = F.mse_loss(feature_rec, feature_align)

        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
            "cls_pred": cls_pred,
            "global_token": global_token,
            "loss_rec": loss_rec,
            "loss_scl": self.lambda_scl * loss_scl_raw,
            "loss_cls": self.lambda_cls * loss_cls_raw,
            "loss_cos": self.lambda_cos * loss_cos_raw,
        }
