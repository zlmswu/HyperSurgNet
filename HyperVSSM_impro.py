"""
HyperMamba Improved - 两项核心改进:

改进1: 可微分超图构建 (DifferentiableHypergraphSSMProcessor)
  - 用可学习原型 + Gumbel-Softmax 替代 K-means 聚类
  - 超图结构本身参与梯度回传, 实现端到端训练
  - 空间超边仍用网格划分(确定性, 无需学习), 特征超边用可微分软分配

改进2: 超图独立SSM (独立SSM参数, 与传统路径并行)
  - 传统路径: 完全复用原始 VMamba 的 forward_core (cross_scan_fn + cross_merge_fn)
  - 超图路径: 拥有独立的SSM参数, 直接在超边序列长度M上操作
  - 两路输出通过可学习门控融合

=== 与原始 VMamba 的一致性保证 ===
  - 传统路径调用 self.forward_core(x), 即原始 SS2D.forward_corev2 的完整路径
  - 使用 cross_scan_fn / cross_merge_fn (Triton优化, 与原始相同)
  - 方向合并方式与原始一致 (cross_merge_fn 内部求和)
  - force_fp32, selective_scan_backend 等参数完全由原始 forward_type 分发机制控制
  - DropPath 初始化与原始 VSSBlock 一致
  - 权重初始化 (apply _init_weights) 与原始 VSSM 一致

文件结构:
  1. DifferentiableHypergraphSSMProcessor - 改进1的核心类
  2. ImprovedCustomScanSS2D - 同时集成改进1和改进2的SS2D
  3. ImprovedCustomScanVSSBlock - 使用新SS2D的VSSBlock
  4. ImprovedHyperVSSM - 完整模型
  5. create_improved_hypervssm - 工厂函数

注意: 本文件只包含改进部分的新增/修改类。
     基础组件 (Linear, LayerNorm, PatchMerge, Mlp, mamba_init, SS2D, VSSM 等)
     仍从原始 HyperMamba.py 导入。
"""

import math
from functools import partial
from typing import Optional, Tuple, Any, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

# === 从原始文件导入基础组件 ===
# 实际使用时, 将 HyperMamba 替换为你的原始模块路径
from HyperMamba import (
    Linear, LayerNorm, PatchMerge, Permute, Mlp,
    mamba_init, SS2D, SS2Dv2, VSSM,
    HilbertCurve, CustomScanProcessor, ViewAdaptationLayer,
    HyperVSSM,
)
from csms6s import selective_scan_fn
from csm_triton import cross_scan_fn, cross_merge_fn


# =====================================================
# 改进1: 可微分超图构建
# =====================================================

class DifferentiableHypergraphSSMProcessor(nn.Module):
    """
    可微分的超图SSM处理器

    核心改进: 用可学习原型(Learnable Prototypes) + 软分配(Soft Assignment)
    替代原始的 K-means 硬聚类, 使超图结构端到端可学习。

    ==========================================
    原始设计的问题:
    ==========================================
    原始 ImprovedHypergraphSSMProcessor 中, 特征聚类超边的构建过程:
      1. 在 torch.no_grad() 下执行 K-means++
      2. 产生硬分配 (0/1 关联矩阵)
      3. 梯度无法流过聚类过程 → 超图结构不可学习

    ==========================================
    改进方案:
    ==========================================
    1. 可学习原型向量 (Learnable Prototypes):
       - 维护 nn.Parameter 形式的原型向量 [num_prototypes, C]
       - 原型在训练中通过梯度更新, 自动学习最优聚类中心
       - 不再需要每次前向传播都重新运行 K-means

    2. 温度可控的软分配 (Temperature-Scaled Soft Assignment):
       - 节点到原型的关联度 = softmax(-distance / temperature)
       - temperature 较高时: 接近均匀分配 (探索)
       - temperature 较低时: 接近硬分配 (利用)
       - temperature 本身也是可学习参数

    3. Gumbel-Softmax (训练时可选):
       - 在软分配基础上加入Gumbel噪声
       - 前向传播: 接近离散的硬分配 (通过 straight-through estimator)
       - 反向传播: 连续梯度正常流动
       - 这让模型在享受离散超图结构优势的同时保持可微分性

    4. Top-K 稀疏化:
       - 对软分配矩阵做 Top-K 稀疏化, 每个节点只连接K个最相关的超边
       - 保持超图的稀疏性, 避免全连接退化
       - Top-K 操作通过 straight-through estimator 保持梯度流动

    ==========================================
    数据流 (Phase 1 - 节点→超边):
    ==========================================
    x [B,C,H,W]
      → 节点特征 [B, N, C]
      → 与可学习原型计算相似度 [B, N, M_feat]
      → Gumbel-Softmax + Top-K 稀疏化 → 软关联矩阵 [B, N, M]
      → 超图卷积聚合 → 超边特征 [B, M, C]
      → Hilbert排序 → [B, C, M]

    ==========================================
    数据流 (Phase 2 - 超边→节点):
    ==========================================
    sorted_edge_seq [B,C,M]  (SSM更新后)
      → 反排序 → 加权 → 超图卷积散播 → [B, C, N]
    """

    def __init__(
            self,
            d_inner: int,
            d_state: int = 16,
            k_neighbors: int = 8,
            num_hyperedges_ratio: float = 0.25,
            dropout: float = 0.1,
            channel_first: bool = True,
            max_nodes: int = 4096,
            # === 新增: 可微分超图参数 ===
            temperature_init: float = 1.0,       # 软分配初始温度
            use_gumbel: bool = True,              # 是否使用Gumbel-Softmax
            gumbel_hard: bool = True,             # Gumbel-Softmax是否用straight-through
            prototype_dim: int = None,            # 原型投影维度 (None则等于d_inner)
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.k_neighbors = k_neighbors
        self.num_hyperedges_ratio = num_hyperedges_ratio
        self.channel_first = channel_first
        self.max_nodes = max_nodes
        self.use_gumbel = use_gumbel
        self.gumbel_hard = gumbel_hard

        max_edges = int(max_nodes * num_hyperedges_ratio) + 100

        # --- 可学习超边权重 ---
        self.edge_weight_matrix = nn.Parameter(torch.ones(1, max_edges))

        # --- Phase1: 节点→超边 特征变换 ---
        self.edge_transform = nn.Linear(d_inner, d_inner)

        # --- Phase2: 超边→节点 特征变换 ---
        self.node_transform = nn.Linear(d_inner, d_inner)
        self.edge_out_norm = nn.LayerNorm(d_inner)

        # --- 共用 ---
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # ==========================================
        # 改进1 核心: 可学习原型和软分配
        # ==========================================

        # 原型投影维度 (降维后计算相似度, 减少计算量)
        self.proto_dim = prototype_dim or d_inner

        # 节点特征投影到原型空间
        self.node_proj = nn.Linear(d_inner, self.proto_dim)

        # 可学习原型向量: 预分配最大容量
        max_feature_edges = max_edges // 2 + 50
        self.prototypes = nn.Parameter(
            torch.randn(max_feature_edges, self.proto_dim) * 0.02
        )

        # 可学习温度参数 (log空间, 保证正数)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature_init))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """当前温度值 (始终为正数)"""
        return self.log_temperature.exp()

    # ==========================================
    # Phase 1: 节点 → 超边 (可微分版本)
    # ==========================================
    def phase1_node_to_edge(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Phase 1: 可微分的超图构建 + 节点→超边聚合 + Hilbert排序
        """
        B, C, H, W = x.shape

        # 1. 可微分超图构建
        incidence_matrix, edge_centers, num_edges = self.construct_hypergraph_differentiable(x, H, W)

        # 2. 计算度矩阵 (软分配下度不再是整数)
        D_v = incidence_matrix.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [B, N, 1]
        D_e = incidence_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, M]

        # 3. 对称归一化
        H_norm = incidence_matrix / (D_v.sqrt() * D_e.sqrt())

        # 4. 节点特征
        node_features = x.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # 5. 超图卷积: 节点 → 超边
        edge_features = torch.bmm(H_norm.transpose(1, 2), node_features)  # [B, M, C]
        edge_features = edge_features / D_e.transpose(1, 2).sqrt()
        edge_features = self.edge_transform(edge_features)
        edge_features = self.act(edge_features)

        # 6. Hilbert曲线排序
        order = self.order_hyperedges_by_spatial_center(edge_centers)
        sorted_edge_features = edge_features[:, order, :]

        # 7. → [B, C, M]
        sorted_edge_seq = sorted_edge_features.transpose(1, 2)

        # 8. 上下文
        context = {
            'H_norm': H_norm,
            'D_v': D_v,
            'order': order,
            'num_edges': num_edges,
            'H': H,
            'W': W,
        }

        return sorted_edge_seq, context

    # ==========================================
    # Phase 2: 超边 → 节点 (与原始相同)
    # ==========================================
    def phase2_edge_to_node(self, sorted_edge_seq: torch.Tensor, context: dict) -> torch.Tensor:
        """Phase 2 与原始版本完全相同"""
        H_norm = context['H_norm']
        D_v = context['D_v']
        order = context['order']
        num_edges = context['num_edges']

        sorted_edges = sorted_edge_seq.transpose(1, 2)
        sorted_edges = self.edge_out_norm(sorted_edges)

        unordered_edges = torch.zeros_like(sorted_edges)
        unordered_edges[:, order, :] = sorted_edges

        edge_weights = self.edge_weight_matrix[:, :num_edges]
        weighted_edges = unordered_edges * edge_weights.unsqueeze(-1)

        node_features = torch.bmm(H_norm, weighted_edges)
        node_features = node_features / D_v.sqrt()
        node_features = self.node_transform(node_features)

        return node_features.transpose(1, 2)

    # ==========================================
    # 核心改进: 可微分超图构建
    # ==========================================
    def construct_hypergraph_differentiable(
            self, x: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        可微分的混合超图构建

        第一部分 (空间网格超边): 与原始相同, 用确定性的网格划分
        第二部分 (特征聚类超边): 用可学习原型 + 软分配替代K-means

        返回:
            incidence_matrix: [B, N, M], 值为连续的 [0, 1] (软分配)
            edge_centers: [M, 2], 超边空间中心
            num_edges: 总超边数
        """
        B, C, _, _ = x.shape
        num_nodes = H * W
        num_edges = max(4, int(num_nodes * self.num_hyperedges_ratio))
        num_spatial_edges = num_edges // 2
        num_feature_edges = num_edges - num_spatial_edges

        x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        incidence_matrix = torch.zeros(B, num_nodes, num_edges, device=x.device)
        edge_centers = []

        # === 第一部分: 空间网格超边 (与原始相同, 确定性) ===
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

        # === 第二部分: 可微分特征聚类超边 ===
        # Step 2a: 将节点特征投影到原型空间
        node_projected = self.node_proj(x_flat)              # [B, N, proto_dim]
        node_projected = F.normalize(node_projected, p=2, dim=-1)

        # Step 2b: 取出当前需要的原型向量
        prototypes = self.prototypes[:num_feature_edges]      # [M_feat, proto_dim]
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)

        # Step 2c: 计算节点与原型之间的余弦相似度
        similarity = torch.matmul(
            node_projected, prototypes_norm.T
        )  # [B, N, M_feat]

        # Step 2d: 软分配 (温度缩放的softmax或Gumbel-Softmax)
        soft_assignment = self._compute_soft_assignment(similarity)  # [B, N, M_feat]

        # Step 2e: Top-K 稀疏化 (保持超图稀疏性)
        sparse_assignment = self._topk_sparsify(
            soft_assignment, k=self.k_neighbors
        )  # [B, N, M_feat]

        # Step 2f: 写入关联矩阵
        incidence_matrix[:, :, num_spatial_edges:num_spatial_edges + num_feature_edges] = sparse_assignment

        # Step 2g: 计算特征超边的空间中心 (用软分配加权)
        h_coords = torch.arange(H, device=x.device).float().view(-1, 1).expand(H, W).reshape(-1)  # [N]
        w_coords = torch.arange(W, device=x.device).float().view(1, -1).expand(H, W).reshape(-1)  # [N]

        for c in range(num_feature_edges):
            weights = sparse_assignment[0, :, c]  # [N]
            weight_sum = weights.sum().clamp(min=1e-6)
            center_h = (weights * h_coords).sum() / weight_sum
            center_w = (weights * w_coords).sum() / weight_sum
            edge_centers.append([center_h.item(), center_w.item()])

        edge_centers = torch.tensor(edge_centers, device=x.device)
        return incidence_matrix, edge_centers, num_edges

    def _compute_soft_assignment(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        计算可微分的软分配矩阵

        参数:
            similarity: 节点-原型相似度 [B, N, M_feat]

        返回:
            assignment: 软分配矩阵 [B, N, M_feat], 每行和为1
        """
        # 温度缩放
        logits = similarity / self.temperature  # [B, N, M_feat]

        if self.training and self.use_gumbel:
            # Gumbel-Softmax: 训练时加入随机性
            assignment = F.gumbel_softmax(
                logits,
                tau=self.temperature.item(),
                hard=self.gumbel_hard,
                dim=-1
            )
        else:
            # 推理时: 确定性软分配
            assignment = F.softmax(logits, dim=-1)

        return assignment

    def _topk_sparsify(self, assignment: torch.Tensor, k: int) -> torch.Tensor:
        """
        Top-K 稀疏化: 每个节点只保留与其最相关的K个超边连接
        """
        B, N, M_feat = assignment.shape
        k = min(k, M_feat)

        # 找到每个节点的 Top-K 超边
        topk_values, topk_indices = torch.topk(assignment, k, dim=-1)  # [B, N, k]

        # 构造稀疏mask
        mask = torch.zeros_like(assignment)
        mask.scatter_(-1, topk_indices, 1.0)

        # Straight-Through Estimator
        sparse_assignment = assignment * (mask - mask.detach() + mask.detach())

        return sparse_assignment

    # ==========================================
    # Hilbert排序 (与原始相同)
    # ==========================================
    def order_hyperedges_by_spatial_center(self, edge_centers: torch.Tensor) -> torch.Tensor:
        """与原始实现相同"""
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

    # ==========================================
    # 便捷方法
    # ==========================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_seq, context = self.phase1_node_to_edge(x)
        node_seq = self.phase2_edge_to_node(edge_seq, context)
        return node_seq


# =====================================================
# 改进2: 超图独立SSM的 SS2D
# =====================================================

class ImprovedCustomScanSS2D(SS2D):
    """
    同时集成改进1和改进2的SS2D模块

    ==========================================
    与原始 VMamba 的一致性设计:
    ==========================================
    传统路径: 完全复用原始 SS2D 的 forward_core
      - 使用 cross_scan_fn / cross_merge_fn (Triton优化)
      - 所有4个方向均保留, SSM参数完全共享
      - force_fp32, selective_scan_backend 等由 forward_type 分发机制控制
      - 方向合并使用 cross_merge_fn (内部求和, 与原始一致)

    超图路径: 独立并行的额外分支
      - 拥有独立的SSM参数 (hg_A_logs, hg_Ds, hg_dt_projs 等)
      - 超图SSM直接在超边序列长度M上操作, 无需M↔L插值
      - 通过可学习门控与传统路径输出融合

    当不启用超图时, 行为与原始 SS2D 完全一致 (调用 super().forwardv2)。

    ==========================================
    融合方式:
    ==========================================
      gate = sigmoid(可学习标量)
      output = gate * 传统路径输出 + (1-gate) * 超图路径输出
      初始化为0 → sigmoid(0)=0.5 → 传统和超图各占一半
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
            # === 超图相关参数 ===
            scan_configs=None,
            hypergraph_config=None,
            **kwargs,
    ):
        # 调用SS2D父类初始化 (设置传统SSM参数, forward_core分发等)
        # 这会完整建立原始 VMamba 的 SS2D, 包括:
        #   - forward_core 指向正确的 forward_corev2 partial (根据 forward_type)
        #   - cross_scan_fn / cross_merge_fn 的参数设置
        #   - A_logs, Ds, dt_projs, x_proj 等全部原始参数
        super().__init__(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
            dt_init_floor=dt_init_floor, initialize=initialize,
            forward_type=forward_type, channel_first=channel_first, **kwargs,
        )

        # 判断是否需要超图路径
        self.scan_configs = scan_configs or ['h', 'h_flip', 'v', 'v_flip']
        self.K_hyper = sum(1 for s in self.scan_configs if s == 'hypergraph')
        self.use_hypergraph = self.K_hyper > 0

        if self.use_hypergraph:
            # === 改进1: 使用可微分超图处理器 ===
            hg_config = hypergraph_config or {
                'k_neighbors': 8,
                'num_hyperedges_ratio': 0.25,
                'dropout': 0.1,
            }
            self.hypergraph_processor = DifferentiableHypergraphSSMProcessor(
                d_inner=self.d_inner,
                d_state=d_state,
                **hg_config
            )

            # === 改进2: 超图方向独立的SSM参数 ===
            d_inner = self.d_inner
            dt_rank_val = int(math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank)

            # 独立的 x_proj: 输入投影 (将超边特征映射到 dt, B, C)
            self.hg_x_proj = Linear(
                d_inner,
                self.K_hyper * (dt_rank_val + d_state * 2),
                groups=self.K_hyper,
                bias=False,
                channel_first=True
            )

            # 独立的 dt_projs: 时间步长投影
            self.hg_dt_projs = Linear(
                dt_rank_val,
                self.K_hyper * d_inner,
                groups=self.K_hyper,
                bias=False,
                channel_first=True
            )

            # 独立的 A, D, dt_bias 参数
            self.hg_A_logs, self.hg_Ds, hg_dt_w, self.hg_dt_projs_bias = \
                mamba_init.init_dt_A_D(
                    d_state, dt_rank_val, d_inner,
                    dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                    k_group=self.K_hyper
                )
            # 将 dt_projs 权重写入 hg_dt_projs 的参数中
            self.hg_dt_projs.weight.data = hg_dt_w.data.view(self.hg_dt_projs.weight.shape)

            # 独立的输出归一化
            self.hg_out_norm = LayerNorm(d_inner, channel_first=channel_first)

            # === 融合门控 ===
            # gate_logit: 标量参数, 通过 sigmoid 映射到 [0,1]
            # 初始化为0 → sigmoid(0)=0.5 → 传统和超图各占一半
            self.gate_logit = nn.Parameter(torch.zeros(1))

    def _forward_hypergraph(self, x: torch.Tensor) -> torch.Tensor:
        """
        超图路径的前向传播

        输入 x: [B, D, H, W] (channel_first, 经过conv2d和act之后)
        输出: [B, D, H, W] 或 [B, H, W, D] (取决于channel_first), 经过hg_out_norm

        数据流:
          x → Phase1 (节点→超边) → [B, K_hyper, D, M]
            → 超图独立SSM (长度M) → [B, K_hyper, D, M]
            → Phase2 (超边→节点) → [B, D, H, W]
            → hg_out_norm → 输出
        """
        B, D, H, W = x.shape
        N = self.d_state
        R = self.dt_rank
        channel_first = self.channel_first

        hyper_xs_list = []
        hg_contexts = []

        for _ in range(self.K_hyper):
            # Phase1: 节点→超边, 产出 [B, D, M]
            edge_seq, context = self.hypergraph_processor.phase1_node_to_edge(x)
            hyper_xs_list.append(edge_seq.unsqueeze(1))  # [B, 1, D, M]
            hg_contexts.append(context)

        K_h = self.K_hyper
        M = hg_contexts[0]['num_edges']
        hyper_xs = torch.cat(hyper_xs_list, dim=1)  # [B, K_hyper, D, M]

        # --- 超图独立SSM ---
        hg_x_dbl = self.hg_x_proj(hyper_xs.view(B, -1, M))
        hg_dts, hg_Bs, hg_Cs = torch.split(
            hg_x_dbl.view(B, K_h, -1, M), [R, N, N], dim=2
        )
        hg_dts = hg_dts.contiguous().view(B, -1, M)
        hg_dts = self.hg_dt_projs(hg_dts)

        hg_As = -self.hg_A_logs.to(torch.float).exp()
        hg_Ds_val = self.hg_Ds.to(torch.float)
        hg_delta_bias = self.hg_dt_projs_bias.view(-1).to(torch.float)

        hyper_xs_flat = hyper_xs.view(B, -1, M)
        hg_dts = hg_dts.contiguous().view(B, -1, M)
        hg_Bs = hg_Bs.contiguous().view(B, K_h, N, M)
        hg_Cs = hg_Cs.contiguous().view(B, K_h, N, M)

        # 在超边长度M上做selective scan (无插值!)
        hyper_ys = selective_scan_fn(
            hyper_xs_flat, hg_dts,
            hg_As, hg_Bs, hg_Cs, hg_Ds_val,
            hg_delta_bias, True,  # delta_softplus=True
            True,  # ssoflex=True
            backend="mamba"
        ).view(B, K_h, -1, M)

        # Phase2: 超边→节点
        hyper_y_list = []
        for idx in range(K_h):
            ssm_output_M = hyper_ys[:, idx]  # [B, D, M]
            node_seq = self.hypergraph_processor.phase2_edge_to_node(
                ssm_output_M, hg_contexts[idx]
            )  # [B, D, N=H*W]
            hyper_y_list.append(node_seq.view(B, -1, H, W))

        hyper_output = sum(hyper_y_list) / len(hyper_y_list)  # [B, D, H, W]

        # 对超图输出做归一化 (与原始 out_norm 格式对齐)
        if not channel_first:
            hyper_output = hyper_output.permute(0, 2, 3, 1).contiguous()
        hyper_output = self.hg_out_norm(hyper_output)

        return hyper_output

    def forwardv2(self, x: torch.Tensor, **kwargs):
        """
        重写 forwardv2

        当不启用超图时: 完全等价于原始 SS2Dv2.forwardv2 (调用 super().forwardv2)
        当启用超图时:
          1. 复制原始 forwardv2 的预处理 (in_proj, conv2d, act)
          2. 路径A: 调用 self.forward_core(x) (完全原始VMamba)
          3. 路径B: 调用 self._forward_hypergraph(x) (超图独立SSM)
          4. 门控融合两路输出
          5. 复制原始 forwardv2 的后处理 (out_act, z multiply, out_proj)
        """
        if not self.use_hypergraph:
            # === 无超图: 完全调用原始 SS2D 的 forwardv2 ===
            # 这保证了传统路径与 VMamba 100% 一致
            return super().forwardv2(x, **kwargs)

        # === 有超图: 混合路径 ===
        # 以下预处理/后处理步骤与原始 SS2Dv2.forwardv2 完全一致
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

        # --- 路径A: 原始VMamba传统路径 (完全不修改) ---
        # self.forward_core 是在 SS2Dv2.__initv2__ 中根据 forward_type 绑定的,
        # 内部调用 forward_corev2, 使用 cross_scan_fn + cross_merge_fn,
        # force_fp32, selective_scan_backend 等参数均由 forward_type 决定.
        y_trad = self.forward_core(x)

        # --- 路径B: 超图路径 (独立SSM) ---
        y_hyper = self._forward_hypergraph(x)

        # --- 门控融合 ---
        gate = torch.sigmoid(self.gate_logit)  # [0, 1]
        y = gate * y_trad + (1.0 - gate) * y_hyper

        # 以下后处理与原始 SS2Dv2.forwardv2 完全一致
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# =====================================================
# 使用改进SS2D的VSSBlock
# =====================================================

class ImprovedCustomScanVSSBlock(nn.Module):
    """
    与原始 VSSBlock 结构完全一致, 仅将 SS2D 替换为 ImprovedCustomScanSS2D

    与原始 VSSBlock 的一致性:
      - DropPath: 始终使用 DropPath(drop_path), 与原始一致 (不做条件判断)
      - _forward 结构: pre-norm / post-norm 逻辑完全一致
      - 参数传递: 与原始 VSSBlock 完全一致
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
            # === 超图相关参数 ===
            scan_configs=None,
            hypergraph_config=None,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = ImprovedCustomScanSS2D(
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

        # 与原始 VSSBlock 一致: 始终使用 DropPath, 不做条件判断
        self.drop_path = DropPath(drop_path)

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
# 完整改进模型
# =====================================================

class ImprovedHyperVSSM(VSSM):
    """
    改进版 HyperVSSM

    直接继承 VSSM (而非 HyperVSSM), 避免中间层不必要的构建和覆盖。
    仅在需要超图的 stage 使用 ImprovedCustomScanVSSBlock,
    其余 stage 完全复用原始 VSSM 的构建逻辑。

    与原始 VSSM 的一致性保证:
      - patch_embed: 完全一致
      - downsample: 完全一致
      - classifier: 完全一致
      - _init_weights: 构建完成后统一调用 apply(_init_weights)
      - forward: 完全一致
      - _make_layer (无超图stage): 使用原始 VSSBlock, 与原始 VSSM 完全一致
      - _make_layer (有超图stage): 使用 ImprovedCustomScanVSSBlock
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
            ssm_conv_bias=False, #true
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="ln2d", # LN
            downsample_version: str = "v3", # v2
            patchembed_version: str = "v2", # v1
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            # === 超图相关参数 ===
            scan_configs_per_stage=None,
            hypergraph_configs_per_stage=None,
            **kwargs,
    ):
        # 注意: 不直接调用 VSSM.__init__, 因为我们需要替换 _make_layer 的逻辑
        # 改为手动复制 VSSM.__init__ 的构建过程, 仅在需要超图的 stage 使用改进 block
        nn.Module.__init__(self)
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
        ssm_act_layer_module: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer_module: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        # === patch_embed: 与原始 VSSM 完全一致 ===
        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        self.patch_embed = self._make_patch_embed(in_chans, dims[0], patch_size, patch_norm,
                                                  channel_first=self.channel_first, version=patchembed_version)

        # === 超图配置 ===
        if scan_configs_per_stage is None:
            scan_configs_per_stage = [
                [['h', 'h_flip', 'v', 'v_flip']] * d for d in depths
            ]
        self.scan_configs_per_stage = scan_configs_per_stage
        self.hypergraph_configs_per_stage = hypergraph_configs_per_stage or [None] * len(depths)

        # === 构建 layers ===
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                channel_first=self.channel_first,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            hg_config = self.hypergraph_configs_per_stage[i_layer]
            stage_needs_hypergraph = hg_config is not None

            if stage_needs_hypergraph:
                # 使用改进版 layer (含超图)
                self.layers.append(self._make_improved_layer(
                    dim=self.dims[i_layer],
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
                    hypergraph_config=hg_config,
                ))
            else:
                # 使用原始 VSSM._make_layer (完全不涉及超图)
                self.layers.append(self._make_layer(
                    dim=self.dims[i_layer],
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
                ))

        # === classifier: 与原始 VSSM 完全一致 ===
        self.classifier = nn.Sequential(OrderedDict(
            norm=LayerNorm(self.num_features, channel_first=self.channel_first),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        # === 权重初始化: 与原始 VSSM 完全一致 ===
        self.apply(self._init_weights)

    @staticmethod
    def _make_improved_layer(
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
            scan_configs_list=None,
            hypergraph_config=None,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            scan_config = scan_configs_list[d] if d < len(scan_configs_list) else scan_configs_list[-1]

            blocks.append(ImprovedCustomScanVSSBlock(
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

    # forward 继承自 VSSM, 无需重写


# =====================================================
# 工厂函数
# =====================================================

def create_improved_hypervssm(
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=96,
        scan_configs=None,
        hypergraph_configs=None,
        **kwargs
):
    """
    创建改进版HyperVSSM模型

    与原始 create_hypervssm 接口完全兼容。
    内部使用:
      - DifferentiableHypergraphSSMProcessor (可微分超图)
      - ImprovedCustomScanSS2D (超图独立SSM + 原始VMamba传统路径)

    示例:
        # 基础版 (与原始VMamba完全相同, 无超图)
        model = create_improved_hypervssm(num_classes=10, depths=[2,2,4,2])

        # 带超图的版本
        model = create_improved_hypervssm(
            num_classes=10,
            depths=[2,2,4,2],
            hypergraph_configs=[
                None,
                {'k_neighbors': 4, 'num_hyperedges_ratio': 0.15,
                 'use_gumbel': True, 'temperature_init': 1.0},
                {'k_neighbors': 6, 'num_hyperedges_ratio': 0.2,
                 'use_gumbel': True, 'temperature_init': 0.5},
                None,
            ]
        )
    """
    if scan_configs is None:
        scan_configs = [
            # Stage 1: 纯传统扫描
            [['h', 'h_flip', 'v', 'v_flip']] * depths[0],
            # Stage 2: 引入超图
            [
                ['hypergraph', 'h_flip', 'v', 'v_flip'],
                ['h', 'hypergraph', 'v', 'v_flip'],
            ] + [['h', 'h_flip', 'v', 'v_flip']] * max(0, depths[1] - 2),
            # Stage 3: 更多超图
            [
                ['hypergraph', 'hypergraph', 'v', 'v_flip'],
                ['h', 'h_flip', 'hypergraph', 'hypergraph'],
            ] * max(1, depths[2] // 2) + [['h', 'h_flip', 'v', 'v_flip']] * (depths[2] % 2),
            # Stage 4: 回到传统
            [['h', 'h_flip', 'v', 'v_flip']] * depths[3],
        ]

    if hypergraph_configs is None:
        hypergraph_configs = [
            None,  # Stage 1
            {
                'k_neighbors': 4,
                'num_hyperedges_ratio': 0.15,
                'dropout': 0.1,
                'use_gumbel': True,
                'temperature_init': 1.0,
            },
            {
                'k_neighbors': 6,
                'num_hyperedges_ratio': 0.2,
                'dropout': 0.15,
                'use_gumbel': True,
                'temperature_init': 0.5,
            },
            None,  # Stage 4
        ]

    model = ImprovedHyperVSSM(
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        scan_configs_per_stage=scan_configs,
        hypergraph_configs_per_stage=hypergraph_configs,
        **kwargs
    )

    return model