import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any, Tuple, List, Union, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from csms6s import *
from csm_triton import cross_scan_fn, cross_merge_fn
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =====================================================
# 基础组件 (原始 model.py)
# =====================================================
class Linear(nn.Linear):
    def __init__(self, *args, channel_first=True, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups

    def forward(self, x: torch.Tensor):
        if self.channel_first:
            if len(x.shape) == 4:
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif len(x.shape) == 3:
                return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)
        else:
            return F.linear(x, self.weight, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        self_state_dict = self.state_dict()
        load_state_dict_keys = list(state_dict.keys())
        if prefix + "weight" in load_state_dict_keys:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view_as(self_state_dict["weight"])
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


class PatchMerge(nn.Module):
    def __init__(self, channel_first=True, in_channel_first=False, out_channel_first=False, ):
        nn.Module.__init__(self)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if not self.in_channel_first:
            B, H, W, C = x.shape

        if (W % 2 != 0) or (H % 2 != 0):
            PH, PW = H - H % 2, W - W % 2
            pad_shape = (PW // 2, PW - PW // 2, PH // 2, PH - PH // 2)
            pad_shape = (*pad_shape, 0, 0, 0, 0) if self.in_channel_first else (0, 0, *pad_shape, 0, 0)
            x = nn.functional.pad(x, pad_shape)

        xs = [
            x[..., 0::2, 0::2], x[..., 1::2, 0::2],
            x[..., 0::2, 1::2], x[..., 1::2, 1::2],
        ] if self.in_channel_first else [
            x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :],
            x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :],
        ]

        xs = torch.cat(xs, (1 if self.out_channel_first else -1))
        return xs


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# =====================================================
# 超图相关组件 (从 hyperVSSM.py 集成)
# =====================================================

class HilbertCurve:
    """
    Hilbert曲线实现类
    相比Z-order曲线具有更好的空间局部性:
    1. 连续性更好 - 曲线上相邻点在空间中也相邻
    2. 局部性更强 - 更好地保持空间聚类结构
    3. 无跳跃 - 曲线不会跨越大距离
    """

    @staticmethod
    def hilbert_d2xy(n: int, d: int):
        """将Hilbert曲线上的距离d转换为2D坐标(x, y)"""
        x = y = 0
        s = 1
        while s < (1 << n):
            rx = 1 & (d >> 1)
            ry = 1 & (d ^ rx)
            x, y = HilbertCurve._rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y

    @staticmethod
    def hilbert_xy2d(n: int, x: int, y: int):
        """将2D坐标(x, y)转换为Hilbert曲线上的距离d"""
        d = 0
        s = (1 << n) >> 1
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = HilbertCurve._rot(s, x, y, rx, ry)
            s >>= 1
        return d

    @staticmethod
    def _rot(n: int, x: int, y: int, rx: int, ry: int):
        """旋转/翻转坐标的辅助函数"""
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y


class CustomScanProcessor:
    """
    自定义扫描处理器类
    支持的扫描模式: h, h_flip, v, v_flip, diag, diag_flip, hypergraph
    """

    @staticmethod
    def apply_scan_pattern(x: torch.Tensor, scan_type: str, hypergraph_processor=None):
        """
        对输入张量应用特定的扫描模式

        参数:
            x: 输入张量, 形状 [B, C, H, W]
            scan_type: 扫描类型字符串
            hypergraph_processor: 超图处理器对象(当scan_type='hypergraph'时需要)

        返回:
            扫描后的序列张量, 形状 [B, C, L], 其中 L=H*W
        """
        B, C, H, W = x.shape
        L = H * W

        if scan_type == 'h':
            return x.flatten(2, 3)

        elif scan_type == 'h_flip':
            return torch.flip(x.flatten(2, 3), dims=[-1])

        elif scan_type == 'v':
            return x.transpose(2, 3).flatten(2, 3)

        elif scan_type == 'v_flip':
            return torch.flip(x.transpose(2, 3).flatten(2, 3), dims=[-1])

        elif scan_type == 'diag':
            result = torch.zeros(B, C, L, device=x.device, dtype=x.dtype)
            idx = 0
            for k in range(H + W - 1):
                for i in range(max(0, k - W + 1), min(k + 1, H)):
                    j = k - i
                    if j < W:
                        result[:, :, idx] = x[:, :, i, j]
                        idx += 1
            return result

        elif scan_type == 'diag_flip':
            result = torch.zeros(B, C, L, device=x.device, dtype=x.dtype)
            idx = 0
            for k in range(H + W - 1, -1, -1):
                for i in range(min(k, H - 1), max(-1, k - W), -1):
                    j = k - i
                    if 0 <= j < W:
                        result[:, :, idx] = x[:, :, i, j]
                        idx += 1
            return result

        elif scan_type == 'hypergraph':
            if hypergraph_processor is None:
                raise ValueError("hypergraph scan requires hypergraph_processor")
            x_hg = hypergraph_processor(x)
            return x_hg

        else:
            raise ValueError(f"Unknown scan type: {scan_type}")

    @staticmethod
    def reverse_scan_pattern(y: torch.Tensor, scan_type: str, H: int, W: int):
        """
        反向恢复扫描模式, 将序列重构为2D结构

        参数:
            y: 扫描后的序列, 形状 [B, C, L]
            scan_type: 扫描类型
            H, W: 目标高度和宽度

        返回:
            重构的2D张量, 形状 [B, C, H, W]
        """
        B, C, L = y.shape

        if scan_type == 'h':
            return y.view(B, C, H, W)

        elif scan_type == 'h_flip':
            return torch.flip(y, dims=[-1]).view(B, C, H, W)

        elif scan_type == 'v':
            return y.view(B, C, W, H).transpose(2, 3)

        elif scan_type == 'v_flip':
            return torch.flip(y, dims=[-1]).view(B, C, W, H).transpose(2, 3)

        elif scan_type in ['diag', 'diag_flip', 'hypergraph']:
            return y.view(B, C, H, W)

        else:
            raise ValueError(f"Unknown scan type: {scan_type}")


class ImprovedHypergraphSSMProcessor(nn.Module):
    """
    改进的超图SSM处理器

    主要特性:
    1. 混合超图构建策略: 结合空间网格和特征聚类
    2. Hilbert曲线空间排序: 比Z-order具有更好的空间局部性
    3. 规范化的超图卷积: 使用标准的超图卷积公式
    4. 可学习的边权重: 为每条超边分配可学习的权重

    超图卷积公式:
    - 节点→超边: Z = D_e^(-1) * H^T * X
    - 超边→节点: Y = D_v^(-1) * H * W * Z
    """

    def __init__(
            self,
            d_inner,
            d_state=16,
            k_neighbors=8,
            num_hyperedges_ratio=0.25,
            dropout=0.1,
            channel_first=True,
            max_nodes=4096,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.k_neighbors = k_neighbors
        self.num_hyperedges_ratio = num_hyperedges_ratio
        self.channel_first = channel_first
        self.max_nodes = max_nodes

        # 预分配边权重矩阵
        max_edges = int(max_nodes * num_hyperedges_ratio) + 100
        self.edge_weight_matrix = nn.Parameter(torch.ones(1, max_edges))

        # 特征转换层
        self.edge_transform = nn.Linear(d_inner, d_inner)
        self.node_transform = nn.Linear(d_inner, d_inner)

        # 归一化和激活函数
        self.norm = nn.LayerNorm(d_inner)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def construct_hypergraph_mixed(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        使用混合策略构建超图(空间网格 + 特征聚类)

        返回:
            incidence_matrix: 关联矩阵 [B, N, M]
            edge_centers: 超边空间中心坐标 [M, 2]
            num_edges: 实际超边数量
        """
        B, C, _, _ = x.shape
        num_nodes = H * W
        num_edges = max(4, int(num_nodes * self.num_hyperedges_ratio))

        num_spatial_edges = num_edges // 2
        num_feature_edges = num_edges - num_spatial_edges

        x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        incidence_matrix = torch.zeros(B, num_nodes, num_edges, device=x.device)
        edge_centers = []

        # === 第一部分: 基于空间网格的超边 ===
        grid_size = int(math.sqrt(num_spatial_edges))
        if grid_size * grid_size < num_spatial_edges:
            grid_size += 1

        edge_idx = 0
        for gh in range(grid_size):
            for gw in range(grid_size):
                if edge_idx >= num_spatial_edges:
                    break

                h_start = gh * H // grid_size
                h_end = (gh + 1) * H // grid_size
                w_start = gw * W // grid_size
                w_end = (gw + 1) * W // grid_size

                center_h = (h_start + h_end) / 2
                center_w = (w_start + w_end) / 2
                edge_centers.append([center_h, center_w])

                for h in range(h_start, h_end):
                    for w in range(w_start, w_end):
                        node_idx = h * W + w
                        incidence_matrix[:, node_idx, edge_idx] = 1.0

                edge_idx += 1

        # === 第二部分: 基于特征聚类的超边 ===
        with torch.no_grad():
            x_norm = F.normalize(x_flat, p=2, dim=-1)

            for b in range(B):
                cluster_centers = self._kmeans_plus_plus_init(x_norm[b], num_feature_edges)

                for _ in range(3):
                    distances = torch.cdist(x_norm[b], cluster_centers)
                    for c in range(num_feature_edges):
                        mask = distances.argmin(dim=1) == c
                        if mask.any():
                            cluster_centers[c] = x_norm[b][mask].mean(0)

                distances = torch.cdist(x_norm[b], cluster_centers)
                for c in range(num_feature_edges):
                    k = min(self.k_neighbors, num_nodes)
                    _, top_indices = torch.topk(distances[:, c], k, largest=False)
                    incidence_matrix[b, top_indices, num_spatial_edges + c] = 1.0

                    h_coords = (top_indices // W).float().mean()
                    w_coords = (top_indices % W).float().mean()
                    if b == 0:
                        edge_centers.append([h_coords.item(), w_coords.item()])

        edge_centers = torch.tensor(edge_centers, device=x.device)
        return incidence_matrix, edge_centers, num_edges

    def _kmeans_plus_plus_init(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """K-means++初始化算法"""
        N, C = x.shape
        centers = []
        centers.append(x[torch.randint(N, (1,)).item()])

        for _ in range(1, k):
            centers_tensor = torch.stack(centers)
            distances = torch.cdist(x, centers_tensor).min(dim=1)[0]
            probabilities = distances ** 2
            probabilities = probabilities / probabilities.sum()
            cumulative = probabilities.cumsum(0)
            r = torch.rand(1, device=x.device)
            idx = (cumulative > r).nonzero()[0].item() if (cumulative > r).any() else N - 1
            centers.append(x[idx])

        return torch.stack(centers)

    def order_hyperedges_by_spatial_center(self, edge_centers: torch.Tensor) -> torch.Tensor:
        """基于空间中心使用Hilbert曲线对超边进行排序"""
        max_coord = max(edge_centers[:, 0].max().item(), edge_centers[:, 1].max().item())
        if max_coord <= 0:
            max_coord = 1.0

        norm_centers = edge_centers / max_coord

        n = max(1, int(np.ceil(np.log2(max_coord + 1))))
        n = min(n, 10)

        hilbert_distances = []
        for center in norm_centers:
            h = int(center[0].item() * ((1 << n) - 1))
            w = int(center[1].item() * ((1 << n) - 1))
            h = max(0, min(h, (1 << n) - 1))
            w = max(0, min(w, (1 << n) - 1))
            d = HilbertCurve.hilbert_xy2d(n, h, w)
            hilbert_distances.append(d)

        order = torch.tensor(hilbert_distances, device=edge_centers.device).argsort()
        return order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        超图序列处理的前向传播(使用标准超图卷积公式)

        参数:
            x: 输入特征图 [B, C, H, W]
        返回:
            处理后的序列 [B, C, N], N=H*W
        """
        B, C, H, W = x.shape

        # 1. 构建混合超图
        incidence_matrix, edge_centers, num_edges = self.construct_hypergraph_mixed(x, H, W)

        # 2. 计算度矩阵
        D_v = incidence_matrix.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [B, N, 1]
        D_e = incidence_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, M]

        # 3. 对称归一化关联矩阵
        H_norm = incidence_matrix / (D_v.sqrt() * D_e.sqrt())

        # 4. 获取边权重
        current_edge_weights = self.edge_weight_matrix[:, :num_edges]

        # 5. 准备节点特征
        node_features = x.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # 6. 超图卷积: 节点→超边
        edge_features = torch.bmm(H_norm.transpose(1, 2), node_features)  # [B, M, C]
        edge_features = edge_features / D_e.transpose(1, 2).sqrt()
        edge_features = self.edge_transform(edge_features)
        edge_features = self.act(edge_features)

        # 7. Hilbert曲线排序
        order = self.order_hyperedges_by_spatial_center(edge_centers)
        ordered_edge_features = edge_features[:, order, :]

        # 8. 处理有序序列
        processed_edges = self.norm(ordered_edge_features)

        # 9. 恢复排序
        unordered_edges = torch.zeros_like(processed_edges)
        unordered_edges[:, order, :] = processed_edges

        # 10. 超图卷积: 超边→节点
        weighted_edges = unordered_edges * current_edge_weights.unsqueeze(-1)
        node_features_updated = torch.bmm(H_norm, weighted_edges)  # [B, N, C]
        node_features_updated = node_features_updated / D_v.sqrt()
        node_features_updated = self.node_transform(node_features_updated)

        # 11. 返回序列
        output_sequence = node_features_updated.transpose(1, 2)  # [B, C, N]
        return output_sequence

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """返回2D输出 [B, C, H, W]"""
        sequence = self.forward(x)
        B, C, N = sequence.shape
        H = W = int(math.sqrt(N))
        return sequence.view(B, C, H, W)


class ViewAdaptationLayer(nn.Module):
    """
    视角适配层 - 将1-3个视角的输入统一转换为3通道
    保留原始存在的通道, 只补充缺失的通道
    """

    def __init__(self):
        super().__init__()

        # 从单视角生成第二个缺失通道
        self.generate_second_from_single = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )

        # 从单视角生成第三个缺失通道
        self.generate_third_from_single = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )

        # 从双视角生成单个缺失通道
        self.generate_single_from_double = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 其中C可以是1, 2, 或3
        Returns:
            [B, 3, H, W] 统一的3通道输出
        """
        B, C, H, W = x.shape

        if C == 3:
            return x
        elif C == 2:
            missing_channel = self.generate_single_from_double(x)
            return torch.cat([x, missing_channel], dim=1)
        elif C == 1:
            missing_channel_1 = self.generate_second_from_single(x)
            missing_channel_2 = self.generate_third_from_single(x)
            return torch.cat([x, missing_channel_1, missing_channel_2], dim=1)
        else:
            raise ValueError(f"Unsupported channel count: {C}, only 1, 2, or 3 are supported")


# =====================================================
# SS2D 原始版本 (v0)
# =====================================================
class SS2Dv0:
    def __initv0__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            dropout=0.0,
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# =====================================================
# SS2D v2 版本
# =====================================================
class SS2Dv2:
    def __initv2__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.forward = self.forwardv2

        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group,
                             bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False,
                               channel_first=True)

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        del self.dt_projs_weight

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            force_fp32=False,
            ssoflex=True,
            selective_scan_backend=None,
            scan_mode="cross2d",
            scan_force_torch=False,
            **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            x_dbl = self.x_proj(xs.view(B, -1, L))
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()
            Ds = self.Ds.to(torch.float)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


# =====================================================
# SS2D 基类 (原始)
# =====================================================
class SS2D(nn.Module, SS2Dv0, SS2Dv2):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
            **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        self_state_dict = self.state_dict()
        self_state_dict_keys = list(self.state_dict().keys())
        load_state_dict_keys = list(state_dict.keys())
        names = {
            "x_proj_weight": "x_proj.weight",
            "x_proj_bias": "x_proj.bias",
            "dt_projs_weight": "dt_projs.weight",
            "dt_projs_bias": "dt_projs.bias",
        }
        for k, v in names.items():
            if (prefix + k in load_state_dict_keys) and (k not in self_state_dict_keys):
                assert v in self_state_dict_keys, f"{v} not in state_dict."
                state_dict[prefix + v] = state_dict[prefix + k].view_as(self_state_dict[v])
                state_dict.pop(prefix + k)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


# =====================================================
# 带超图自定义扫描的 SS2D (从 hyperVSSM.py 集成)
# =====================================================
class CustomScanSS2D(SS2D):
    """
    带有自定义扫描配置和超图的SS2D模块
    支持: h, h_flip, v, v_flip, diag, diag_flip, hypergraph 扫描模式
    """

    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v05_noz",
            channel_first=False,
            scan_configs=None,
            hypergraph_config=None,
            **kwargs,
    ):
        super().__init__(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
            dt_init_floor=dt_init_floor, initialize=initialize,
            forward_type=forward_type, channel_first=channel_first, **kwargs,
        )

        self.scan_configs = scan_configs or ['h', 'h_flip', 'v', 'v_flip']

        self.use_hypergraph = 'hypergraph' in self.scan_configs
        if self.use_hypergraph:
            hg_config = hypergraph_config or {
                'k_neighbors': 8,
                'num_hyperedges_ratio': 0.25,
                'dropout': 0.1
            }
            self.hypergraph_processor = ImprovedHypergraphSSMProcessor(
                d_inner=self.d_inner,
                d_state=d_state,
                **hg_config
            )

    def forward_corev2_custom(
            self,
            x: torch.Tensor,
            force_fp32=False,
            ssoflex=True,
            selective_scan_backend="mamba",
            **kwargs,
    ):
        """使用自定义扫描模式的前向传播"""
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K = len(self.scan_configs)
        D_inner = self.d_inner
        R = self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        # 应用自定义扫描模式
        xs_list = []
        for scan_type in self.scan_configs:
            xs_i = CustomScanProcessor.apply_scan_pattern(
                x, scan_type,
                self.hypergraph_processor if self.use_hypergraph else None
            )
            xs_list.append(xs_i.unsqueeze(1))

        xs = torch.cat(xs_list, dim=1)  # [B, K, C, L]

        # SSM处理流程
        x_dbl = self.x_proj(xs.view(B, -1, L))
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = dts.contiguous().view(B, -1, L)
        dts = self.dt_projs(dts)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        # 执行选择性扫描
        ys = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)

        # 自定义合并
        y_list = []
        for i, scan_type in enumerate(self.scan_configs):
            y_i = CustomScanProcessor.reverse_scan_pattern(
                ys[:, i], scan_type, H, W
            )
            y_list.append(y_i)

        y = sum(y_list) / len(y_list)

        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        """重写forwardv2以使用自定义扫描"""
        x = self.in_proj(x)

        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.with_dconv:
            x = self.conv2d(x)

        x = self.act(x)
        y = self.forward_corev2_custom(x, force_fp32=(not self.disable_force32))
        y = self.out_act(y)

        if not self.disable_z:
            y = y * z

        out = self.dropout(self.out_proj(y))
        return out


# =====================================================
# VSSBlock (原始)
# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            channel_first=False,
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channel_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# =====================================================
# 带超图自定义扫描的 VSSBlock (从 hyperVSSM.py 集成)
# =====================================================
class CustomScanVSSBlock(nn.Module):
    """
    带有自定义扫描配置的VSS块
    结构: LayerNorm → SSM → DropPath → 残差 + LayerNorm → MLP → DropPath → 残差
    """

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            channel_first=False,
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            use_checkpoint: bool = False,
            post_norm: bool = False,
            scan_configs=None,
            hypergraph_config=None,
            **kwargs,
    ):
        nn.Module.__init__(self)

        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = CustomScanSS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
                scan_configs=scan_configs,
                hypergraph_config=hypergraph_config,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channel_first=channel_first
            )

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# =====================================================
# VSSM (原始)
# =====================================================
class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v2",
            patchembed_version: str = "v1",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            _SS2D=SS2D,
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        self.patch_embed = self._make_patch_embed(in_chans, dims[0], patch_size, patch_norm,
                                                  channel_first=self.channel_first, version=patchembed_version)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                channel_first=self.channel_first,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                downsample=downsample,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                _SS2D=_SS2D,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=LayerNorm(self.num_features, channel_first=self.channel_first),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, channel_first=False, version="v1"):
        if version == "v1":
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first)
                 if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        elif version == "v2":
            stride = patch_size // 2
            kernel_size = stride + 1
            padding = 1
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim // 2, channel_first=True) if patch_norm else nn.Identity()),
                nn.Identity(),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first)
                 if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        raise NotImplementedError

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm=True, channel_first=False, version="v1"):
        if version == "v1":
            return nn.Sequential(
                PatchMerge(channel_first),
                LayerNorm(4 * dim, channel_first=channel_first) if norm else nn.Identity(),
                Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False, channel_first=channel_first),
            )
        elif version == "v2":
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif version == "v3":
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        raise NotImplementedError

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            downsample=nn.Identity(),
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4],
                                                             align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
            change_name(f"layers.{i}.downsample.norm", f"layers.{i}.downsample.{1}")
            change_name(f"layers.{i}.downsample.reduction", f"layers.{i}.downsample.{2}")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


# =====================================================
# 带超图的完整 VSSM (从 hyperVSSM.py 集成)
# =====================================================
class HyperVSSM(VSSM):
    """
    带有自定义扫描配置和超图的完整VSSM模型

    特性:
    - 每个stage可以有不同的扫描配置(h/v/diag/hypergraph等)
    - 集成改进的超图处理器(混合超图+Hilbert排序)
    - 视角适配层(支持1-3通道输入)
    - 灵活的超图配置
    """

    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 4, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v2",
            patchembed_version: str = "v1",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            # === 超图相关新参数 ===
            scan_configs_per_stage=None,
            hypergraph_configs_per_stage=None,
            use_view_adaptation=False,
            **kwargs,
    ):
        # 调用父类VSSM的初始化
        super().__init__(
            patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            depths=depths, dims=dims, ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank, ssm_act_layer=ssm_act_layer, ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias, ssm_drop_rate=ssm_drop_rate, ssm_init=ssm_init,
            forward_type=forward_type, mlp_ratio=mlp_ratio, mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate, gmlp=gmlp, drop_path_rate=drop_path_rate,
            patch_norm=patch_norm, norm_layer=norm_layer, downsample_version=downsample_version,
            patchembed_version=patchembed_version, use_checkpoint=use_checkpoint,
            posembed=posembed, imgsize=imgsize, **kwargs,
        )

        # 设置默认扫描配置
        if scan_configs_per_stage is None:
            scan_configs_per_stage = [
                [['h', 'h_flip', 'v', 'v_flip']] * d for d in depths
            ]

        self.scan_configs_per_stage = scan_configs_per_stage
        self.hypergraph_configs_per_stage = hypergraph_configs_per_stage or [None] * len(depths)

        # 重建所有层 (使用自定义扫描block)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        ssm_act_layer_module = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        mlp_act_layer_module = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1] if i_layer < self.num_layers - 1 else -1,
                channel_first=self.channel_first,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_hyper_layer(
                dim=self.dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                downsample=downsample,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer_module,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer_module,
                mlp_drop_rate=mlp_drop_rate,
                scan_configs_list=self.scan_configs_per_stage[i_layer],
                hypergraph_config=self.hypergraph_configs_per_stage[i_layer],
            ))

        # 视角适配层 (可选)
        self.use_view_adaptation = use_view_adaptation
        if use_view_adaptation:
            self.view_adaptation = ViewAdaptationLayer()

    @staticmethod
    def _make_hyper_layer(
            dim=96,
            depth=2,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            downsample=nn.Identity(),
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            scan_configs_list=None,
            hypergraph_config=None,
            **kwargs,
    ):
        """创建带有超图自定义扫描的stage层"""
        blocks = []
        for d in range(depth):
            scan_config = scan_configs_list[d] if d < len(scan_configs_list) else scan_configs_list[-1]

            blocks.append(CustomScanVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                scan_configs=scan_config,
                hypergraph_config=hypergraph_config,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks),
            downsample=downsample,
        ))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取空间特征(在全局池化之前)"""
        if self.use_view_adaptation:
            x = self.view_adaptation(x)

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.classifier.norm(x)
        return x

    def forward(self, x: torch.Tensor, return_spatial_features: bool = False):
        """
        前向传播, 支持返回空间特征

        Args:
            x: 输入图像 [B, C, H, W]
            return_spatial_features: 是否返回空间特征图

        Returns:
            分类logits [B, num_classes] 或 空间特征图
        """
        if return_spatial_features:
            return self.forward_features(x)
        else:
            x = self.forward_features(x)
            if not self.channel_first:
                x = x.permute(0, 3, 1, 2)
            x = self.classifier.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier.head(x)
            return x


# =====================================================
# Backbone (兼容 openmmlab)
# =====================================================
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["ln2d"])

        self.out_indices = out_indices
        for i in out_indices:
            layer = LayerNorm(self.dims[i], channel_first=self.channel_first)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        return outs


# =====================================================
# HyperVSSM Backbone (兼容 openmmlab, 带超图)
# =====================================================
class Backbone_HyperVSSM(HyperVSSM):
    """带超图的Backbone, 兼容openmmlab检测/分割框架"""

    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["ln2d"])

        self.out_indices = out_indices
        for i in out_indices:
            layer = LayerNorm(self.dims[i], channel_first=self.channel_first)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        if self.use_view_adaptation:
            x = self.view_adaptation(x)

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        return outs


# =====================================================
# 工厂函数
# =====================================================
def create_hypervssm(
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=96,
        scan_configs=None,
        hypergraph_configs=None,
        **kwargs
):
    """
    创建HyperVSSM模型的工厂函数

    默认扫描策略:
    - Stage 1: 纯传统扫描(建立基础特征)
    - Stage 2: 逐步引入超图(开始捕获高阶关系)
    - Stage 3: 更多超图扫描(充分利用超图能力)
    - Stage 4: 回到传统扫描(稳定特征)
    """
    if scan_configs is None:
        scan_configs = [
            # Stage 1: 纯传统扫描
            [['h', 'h_flip', 'v', 'v_flip']] * depths[0],
            # Stage 2: 逐步引入超图
            [
                ['hypergraph', 'h_flip', 'v', 'v_flip'],
                ['h', 'hypergraph', 'v', 'v_flip'],
            ] + [['h', 'h_flip', 'v', 'v_flip']] * max(0, depths[1] - 2),
            # Stage 3: 更多超图扫描
            [
                ['hypergraph', 'hypergraph', 'v', 'v_flip'],
                ['h', 'h_flip', 'hypergraph', 'hypergraph'],
            ] * max(1, depths[2] // 2) + [['h', 'h_flip', 'v', 'v_flip']] * (depths[2] % 2),
            # Stage 4: 回到传统扫描
            [['h', 'h_flip', 'v', 'v_flip']] * depths[3],
        ]

    if hypergraph_configs is None:
        hypergraph_configs = [
            None,  # Stage 1
            {'k_neighbors': 4, 'num_hyperedges_ratio': 0.15, 'dropout': 0.1},  # Stage 2
            {'k_neighbors': 6, 'num_hyperedges_ratio': 0.2, 'dropout': 0.15},  # Stage 3
            None,  # Stage 4
        ]

    model = HyperVSSM(
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        scan_configs_per_stage=scan_configs,
        hypergraph_configs_per_stage=hypergraph_configs,
        **kwargs
    )

    return model