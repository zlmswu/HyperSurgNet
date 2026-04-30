# improved_medical_gnn_model.py
"""
改进的医疗GNN模型
集成：动态图学习 + 改进损失函数 + 可解释性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ==================== 动态图学习模块 ====================
class AdaptiveGraphLearner(nn.Module):
    """自适应动态图学习模块"""

    def __init__(self, hidden_dim, k_neighbors=10, temperature=0.5):
        super().__init__()
        self.k = k_neighbors
        self.temperature = temperature

        # 节点相似度计算网络
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 边权重学习网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 静态图先验权重（可学习）
        self.prior_weight = nn.Parameter(torch.tensor(0.3))

    def compute_similarity(self, node_features):
        """计算节点间相似度矩阵"""
        batch_size, num_nodes, _ = node_features.shape

        # 投影到相似度空间
        projected = self.similarity_net(node_features)

        # 计算余弦相似度
        normalized = F.normalize(projected, p=2, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))

        return similarity

    def sparse_graph_construction(self, similarity, static_edges=None, num_nodes=37):
        """构建稀疏图（Top-K + 静态先验）"""
        batch_size = similarity.shape[0]

        # Top-K稀疏化
        topk_values, topk_indices = torch.topk(similarity, k=self.k, dim=-1)

        # 创建稀疏邻接矩阵
        mask = torch.zeros_like(similarity)
        mask.scatter_(-1, topk_indices, 1.0)

        # 对称化
        mask = (mask + mask.transpose(1, 2)) / 2.0

        # Gumbel-Softmax采样（可微分）
        sparse_adj = F.gumbel_softmax(
            similarity / self.temperature,
            tau=1.0,
            hard=False,
            dim=-1
        ) * mask

        # 融合静态先验图
        if static_edges is not None:
            static_adj = self._edges_to_adj(static_edges, num_nodes, batch_size)
            alpha = torch.sigmoid(self.prior_weight)
            sparse_adj = alpha * sparse_adj + (1 - alpha) * static_adj

        return sparse_adj

    def forward(self, node_features, static_edges=None):
        """
        Args:
            node_features: [batch, num_nodes, hidden_dim]
            static_edges: [2, num_edges] 静态图边索引
        Returns:
            edge_index: [2, num_edges] 动态图边索引
            edge_weights: [num_edges] 边权重
            adj_matrix: [batch, num_nodes, num_nodes] 邻接矩阵
        """
        batch_size, num_nodes, hidden_dim = node_features.shape

        # 计算相似度
        similarity = self.compute_similarity(node_features)

        # 构建稀疏图
        adj_matrix = self.sparse_graph_construction(similarity, static_edges, num_nodes)

        # 转换为边列表格式（取批次平均）
        adj_mean = adj_matrix.mean(dim=0)
        edge_index, edge_weights = self._adj_to_edges(adj_mean)

        return edge_index, edge_weights, adj_matrix

    def _adj_to_edges(self, adj_matrix):
        """将邻接矩阵转换为边索引和权重"""
        threshold = 0.1
        mask = adj_matrix > threshold
        edge_index = mask.nonzero().t()
        edge_weights = adj_matrix[edge_index[0], edge_index[1]]

        return edge_index, edge_weights

    def _edges_to_adj(self, edges, num_nodes, batch_size):
        """将边索引转换为邻接矩阵"""
        adj = torch.zeros(batch_size, num_nodes, num_nodes, device=edges.device)
        adj[:, edges[0], edges[1]] = 1.0
        return adj


# ==================== 改进的多尺度GNN ====================
class ImprovedMultiScaleGNN(nn.Module):
    """改进的多尺度GNN（集成动态图学习）"""

    def __init__(self, hidden_dim, heads=4, dropout=0.2, k_neighbors=10):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 动态图学习器
        self.graph_learner = AdaptiveGraphLearner(
            hidden_dim,
            k_neighbors=k_neighbors
        )

        # 多尺度GATv2层（更强的注意力机制）
        self.local_gat = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        ])

        self.mid_gat = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        ])

        self.global_gat = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        ])

        # 尺度融合注意力
        self.scale_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # 可学习的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(3))

    def forward(self, x, static_edge_index=None):
        """
        Args:
            x: [batch, num_nodes, hidden_dim]
            static_edge_index: [2, num_edges] 静态图
        Returns:
            enhanced_features: [batch, num_nodes, hidden_dim]
            dynamic_edges: 动态学习的边
            edge_weights: 边权重
            attention_weights: 注意力权重（用于可解释性）
        """
        batch_size, num_nodes, hidden_dim = x.shape

        # 动态学习图结构
        dynamic_edges, edge_weights, adj_matrix = self.graph_learner(
            x, static_edge_index
        )

        # 展平用于GNN
        x_flat = x.reshape(-1, hidden_dim)

        # 为批处理调整边索引
        batch_edges = self._create_batch_edges(dynamic_edges, batch_size, num_nodes)

        # 多尺度传播（保存注意力权重用于可解释性）
        local_out, local_att = self.local_gat[0](x_flat, batch_edges, return_attention_weights=True)
        local_out = self.local_gat[1](local_out)
        local_out = self.local_gat[2](local_out)

        mid_out, mid_att = self.mid_gat[0](local_out, batch_edges, return_attention_weights=True)
        mid_out = self.mid_gat[1](mid_out)
        mid_out = self.mid_gat[2](mid_out)

        global_out, global_att = self.global_gat[0](mid_out, batch_edges, return_attention_weights=True)
        global_out = self.global_gat[1](global_out)
        global_out = self.global_gat[2](global_out)

        # 重塑为批次格式
        local_out = local_out.reshape(batch_size, num_nodes, -1)
        mid_out = mid_out.reshape(batch_size, num_nodes, -1)
        global_out = global_out.reshape(batch_size, num_nodes, -1)

        # 尺度融合
        scale_stack = torch.stack([local_out, mid_out, global_out], dim=2)
        scale_stack = scale_stack.reshape(batch_size * num_nodes, 3, -1)

        fused, fusion_weights = self.scale_fusion(scale_stack, scale_stack, scale_stack)
        fused = fused.mean(dim=1).reshape(batch_size, num_nodes, -1)

        # 收集注意力权重用于可解释性
        attention_weights = {
            'local': local_att[1].detach(),
            'mid': mid_att[1].detach(),
            'global': global_att[1].detach(),
            'fusion': fusion_weights.detach()
        }

        return fused, dynamic_edges, edge_weights, attention_weights

    def _create_batch_edges(self, edge_index, batch_size, num_nodes):
        """为批处理创建边索引"""
        edge_list = []
        for b in range(batch_size):
            offset = b * num_nodes
            batch_edges = edge_index + offset
            edge_list.append(batch_edges)
        return torch.cat(edge_list, dim=1)


# ==================== 主模型 ====================
class ImprovedMedicalGNN(nn.Module):
    """改进的医疗GNN模型（完整版）"""

    def __init__(self,
                 num_numerical_features: int = 23,
                 categorical_dims: Dict[str, int] = None,
                 hidden_dim: int = 128,
                 heads: int = 4,
                 dropout: float = 0.2,
                 num_classes: int = 2,
                 k_neighbors: int = 10):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_numerical = num_numerical_features
        self.categorical_dims = categorical_dims or {}

        # 特征编码层
        self.feature_encoder = FeatureEncoder(
            num_numerical_features,
            categorical_dims,
            hidden_dim
        )

        # 改进的多尺度特征提取（集成动态图学习）
        self.multi_scale_extractor = ImprovedMultiScaleGNN(
            hidden_dim,
            heads,
            dropout,
            k_neighbors
        )

        # 层次化特征聚合
        self.hierarchical_aggregator = HierarchicalFeatureAggregator(
            hidden_dim,
            dropout
        )

        # 残差连接
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)

        # 诊断决策头
        self.diagnostic_head = DiagnosticHead(
            hidden_dim,
            dropout,
            num_classes
        )

        # 特征图构建器（提供静态先验）
        self.graph_builder = MedicalGraphBuilder()

        self.apply(self._init_weights)

    def forward(self, numerical_data, categorical_data, edge_index=None, batch=None):
        """
        前向传播
        Returns:
            包含logits、embeddings和可解释性信息的字典
        """
        # 1. 特征编码
        node_features, feature_masks = self.feature_encoder(numerical_data, categorical_data)

        # 2. 构建静态先验图
        if edge_index is None:
            edge_index = self.graph_builder.build_edges().to(node_features.device)

        # 3. 动态多尺度特征提取
        multi_scale_output, dynamic_edges, edge_weights, attention_weights = \
            self.multi_scale_extractor(node_features, edge_index)

        # 4. 残差连接
        residual = self.residual_projection(node_features)
        enhanced_features = multi_scale_output + residual

        # 5. 层次化特征聚合
        hierarchical_output, group_attentions = self.hierarchical_aggregator(
            enhanced_features, dynamic_edges, feature_masks
        )

        # 6. 最终诊断预测
        diagnosis_logits = self.diagnostic_head(hierarchical_output)

        return {
            'logits': diagnosis_logits,
            'node_embeddings': node_features,
            'multi_scale_embeddings': multi_scale_output,
            'hierarchical_embeddings': hierarchical_output,
            'dynamic_edges': dynamic_edges,
            'edge_weights': edge_weights,
            'attention_weights': attention_weights,
            'group_attentions': group_attentions,
            'static_edges': edge_index
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)


# ==================== 辅助模块 ====================
class FeatureEncoder(nn.Module):
    """特征编码器"""

    def __init__(self, num_numerical, categorical_dims, hidden_dim):
        super().__init__()

        self.num_numerical = num_numerical
        self.hidden_dim = hidden_dim

        # 数值特征编码器
        self.numerical_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_numerical)
        ])

        # 分类特征嵌入
        self.categorical_embeddings = nn.ModuleDict()
        for feature_name, num_classes in categorical_dims.items():
            self.categorical_embeddings[feature_name] = nn.Embedding(
                num_classes + 1,
                hidden_dim
            )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 特征掩码
        '''self.feature_masks = {
            'clinical': list(range(0, 7)),  # 0-4\23-26
            'physical': list(range(7, 15)), # 20-23\32-37
            'laboratory': list(range(15, 37)) # 4-20\26-32
        }'''
        # 特征掩码
        self.feature_masks = {
            'clinical': (
                    list(range(0, 4)) +
                    list(range(23, 26))
            ),

            'physical': (
                    list(range(20, 23)) +
                    list(range(32, 37))
            ),

            'laboratory': (
                    list(range(4, 20)) +
                    list(range(26, 32))
            )
        }

    def forward(self, numerical_data, categorical_data):
        encoded_features = []
        batch_size = numerical_data.shape[0]

        # 编码数值特征
        if numerical_data is not None:
            for i, encoder in enumerate(self.numerical_encoders):
                if i < numerical_data.shape[1]:
                    num_val = numerical_data[:, i:i + 1]
                    encoded_num = encoder(num_val)
                    encoded_features.append(encoded_num.unsqueeze(1))

        # 编码分类特征
        for feature_name, embedding_layer in self.categorical_embeddings.items():
            if feature_name in categorical_data:
                cat_val = categorical_data[feature_name]
                if cat_val.dim() == 0:
                    cat_val = cat_val.unsqueeze(0)
                embedded_cat = embedding_layer(cat_val.long())
                encoded_features.append(embedded_cat.unsqueeze(1))

        # 拼接所有特征
        if encoded_features:
            node_features = torch.cat(encoded_features, dim=1)
            batch_size, num_nodes, _ = node_features.shape
            node_features = node_features.reshape(-1, self.hidden_dim)
            node_features = self.feature_fusion(node_features)
            node_features = node_features.reshape(batch_size, num_nodes, self.hidden_dim)
        else:
            node_features = torch.zeros(batch_size, 37, self.hidden_dim, device=numerical_data.device)

        return node_features, self.feature_masks


class HierarchicalFeatureAggregator(nn.Module):
    """层次化特征聚合模块"""

    def __init__(self, hidden_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 分组注意力池化
        self.group_poolers = nn.ModuleDict({
            'clinical': GroupAttentionPooling(hidden_dim, dropout=dropout),
            'physical': GroupAttentionPooling(hidden_dim, dropout=dropout),
            'laboratory': GroupAttentionPooling(hidden_dim, dropout=dropout)
        })

        # 层次特征融合
        self.hierarchy_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_index, feature_masks):
        batch_size = node_features.shape[0]
        group_representations = []
        group_attention_scores = {}

        # 组内聚合
        for group_name, node_indices in feature_masks.items():
            group_features = node_features[:, node_indices, :]
            pooled_features, attention_scores = self.group_poolers[group_name](group_features)
            group_representations.append(pooled_features)
            group_attention_scores[group_name] = attention_scores

        # 跨层次交互
        hierarchy_tensor = torch.stack(group_representations, dim=1)
        attended, _ = self.hierarchy_fusion(hierarchy_tensor, hierarchy_tensor, hierarchy_tensor)
        fused = attended.mean(dim=1)
        output = self.final_norm(fused)

        return output, group_attention_scores


class GroupAttentionPooling(nn.Module):
    """分组注意力池化"""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch_size, num_nodes, hidden_dim = x.shape

        attention_scores = self.attention_net(x).squeeze(-1)
        weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)

        aggregated = torch.sum(x * weights, dim=1)
        output = self.output_projection(aggregated)

        return output, attention_scores


class DiagnosticHead(nn.Module):
    """诊断决策头"""

    def __init__(self, hidden_dim, dropout, num_classes=2):
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class MedicalGraphBuilder:
    """医学特征图构建器"""

    def __init__(self):
        # 定义医学逻辑连接
        self.medical_connections = [
            (15, 18), (20, 26), (7, 15),  # 炎症指标
            (21, 22), (23, 24), (25, 21),  # 电解质
            (28, 29), (30, 28),  # 肾功能
            (10, 32), (12, 33), (13, 34),  # 临床症状
            (9, 8), (7, 9),  # 生命体征
            (4, 11), (11, 12),  # 疼痛
        ]

        self.feature_masks = {
            'clinical': list(range(0, 7)),
            'physical': list(range(7, 15)),
            'laboratory': list(range(15, 37))
        }

    def build_edges(self, feature_masks=None):
        """构建特征图边关系"""
        if feature_masks is None:
            feature_masks = self.feature_masks

        edges = []

        # 组内链式连接
        for group_name, indices in feature_masks.items():
            for i in range(len(indices) - 1):
                edges.append([indices[i], indices[i + 1]])
                edges.append([indices[i + 1], indices[i]])

        # 医学逻辑连接
        for src_idx, dst_idx in self.medical_connections:
            edges.append([src_idx, dst_idx])
            edges.append([dst_idx, src_idx])

        # 自环
        for i in range(37):
            edges.append([i, i])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index


# ==================== 数据集类 ====================
class MedicalGraphDataset(Dataset):
    """医疗图数据集"""

    def __init__(self, numerical_data, categorical_data, labels, edge_index):
        self.numerical_data = numerical_data
        self.categorical_data = categorical_data
        self.labels = labels
        self.edge_index = edge_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'numerical_features': self.numerical_data[idx],
            'categorical_features': {
                key: val[idx] for key, val in self.categorical_data.items()
            },
            'edge_index': self.edge_index,
            'y': self.labels[idx]
        }


# ==================== 可解释性模块 ====================
class GNNExplainer:
    """GNN可解释性分析器"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def explain_prediction(self, batch_data, feature_names):
        """解释单个预测"""
        self.model.eval()

        numerical_features = batch_data['numerical_features'].to(self.device)
        categorical_features = {
            k: v.to(self.device) for k, v in batch_data['categorical_features'].items()
        }

        # 需要梯度
        numerical_features.requires_grad = True

        # 前向传播
        outputs = self.model(numerical_features, categorical_features)
        logits = outputs['logits']

        # 反向传播获取梯度
        logits.sum().backward()

        # 特征重要性
        feature_importance = numerical_features.grad.abs().mean(dim=0).cpu().numpy()

        # 获取预测
        with torch.no_grad():
            prob = torch.sigmoid(logits).cpu().numpy()
            prediction = (prob > 0.5).astype(int)

        # 注意力权重
        attention_weights = outputs['attention_weights']

        return {
            'feature_importance': feature_importance,
            'attention_weights': attention_weights,
            'dynamic_edges': outputs['dynamic_edges'].cpu(),
            'edge_weights': outputs['edge_weights'].cpu(),
            'prediction': prediction,
            'confidence': prob
        }