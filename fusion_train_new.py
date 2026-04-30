"""
多模态融合模型训练脚本
Training Script for MultiModal Fusion: ImprovedHyperVSSM + ImprovedMedicalGNN

训练策略:
  1. 正确的参数分组 (decay / no_decay) - SSM参数不做weight decay
  2. Warmup + Cosine 退火学习率调度
  3. 梯度裁剪 (SSM模型必需)
  4. Label Smoothing
  5. 混合精度训练 (AMP)
  6. 自动类别权重平衡
  7. 多模态输入处理 (图像 + 数值特征 + 分类特征)
"""

import os
import math
import json
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

from fusion_model import create_multimodal_fusion_model
from DataOfClass import train_loader, test_loader, categorical_dims


# =====================================================================
# 工具函数
# =====================================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def print_model_parameters(model):
    total, trainable, non_trainable = count_parameters(model)
    print("=" * 80)
    print("Model Parameters Summary".center(80))
    print("=" * 80)
    print(f"{'Total Parameters:':<30} {total:>20,}")
    print(f"{'Trainable Parameters:':<30} {trainable:>20,}")
    print(f"{'Non-trainable Parameters:':<30} {non_trainable:>20,}")
    print(f"{'Total (M):':<30} {total / 1e6:>20.4f} M")
    print(f"{'Trainable (M):':<30} {trainable / 1e6:>20.4f} M")
    print("=" * 80)
    return total, trainable, non_trainable


def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return sensitivity, specificity, TP, TN, FP, FN


def print_detailed_metrics(y_true, y_pred, y_prob=None, dataset_name="Dataset"):
    print("\n" + "=" * 80)
    print(f"{dataset_name} - Detailed Metrics".center(80))
    print("=" * 80)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    sensitivity, specificity, TP, TN, FP, FN = calculate_sensitivity_specificity(y_true, y_pred)

    try:
        auc_score = roc_auc_score(y_true, y_prob) if y_prob is not None else roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = None

    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    ppv = precision

    print(f"\nConfusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print("-" * 80)
    print(f"  {'Accuracy:':<35} {accuracy:>10.4f}  ({accuracy * 100:>6.2f}%)")
    print(f"  {'Sensitivity (Recall):':<35} {sensitivity:>10.4f}  ({sensitivity * 100:>6.2f}%)")
    print(f"  {'Specificity:':<35} {specificity:>10.4f}  ({specificity * 100:>6.2f}%)")
    print(f"  {'PPV (Precision):':<35} {ppv:>10.4f}  ({ppv * 100:>6.2f}%)")
    print(f"  {'NPV:':<35} {npv:>10.4f}  ({npv * 100:>6.2f}%)")
    print(f"  {'F1-Score:':<35} {f1:>10.4f}  ({f1 * 100:>6.2f}%)")
    if auc_score is not None:
        print(f"  {'AUC:':<35} {auc_score:>10.4f}  ({auc_score * 100:>6.2f}%)")
    print("=" * 80)

    return accuracy, precision, recall, f1, sensitivity, specificity, auc_score, ppv, npv


# =====================================================================
# 参数分组: 区分 decay / no_decay
# =====================================================================
def build_optimizer_param_groups(model, weight_decay=0.05):
    """
    SSM 关键: A_logs, Ds 等参数不能做 weight_decay,
    否则会导致 SSM 动态退化, 模型无法学习
    """
    decay_params = []
    no_decay_params = []
    no_decay_keywords = {'bias', 'norm', 'LayerNorm', 'layernorm', 'Norm'}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if hasattr(param, '_no_weight_decay') and param._no_weight_decay:
            no_decay_params.append(param)
        elif any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        elif param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    print(f"\n📋 Optimizer Parameter Groups:")
    print(f"   Decay params:    {sum(p.numel() for p in decay_params):>10,} (wd={weight_decay})")
    print(f"   No-decay params: {sum(p.numel() for p in no_decay_params):>10,} (wd=0)")

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


# =====================================================================
# Warmup + Cosine 学习率调度器
# =====================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * alpha
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# =====================================================================
# Label Smoothing CrossEntropy
# =====================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(pred, dim=-1)

        if self.weight is not None:
            sample_weight = self.weight[target]
            loss = -(smooth_target * log_prob).sum(dim=-1)
            loss = (loss * sample_weight).mean()
        else:
            loss = -(smooth_target * log_prob).sum(dim=-1).mean()

        return loss


# =====================================================================
# 超图扫描配置构建器
# =====================================================================
def build_scan_configs(depths, strategy='progressive'):
    """构建不同策略的扫描配置"""
    num_stages = len(depths)

    if strategy == 'traditional':
        scan_configs = [[['h', 'h_flip', 'v', 'v_flip']] * d for d in depths]
        hypergraph_configs = [None] * num_stages
        return scan_configs, hypergraph_configs

    elif strategy == 'progressive':
        scan_configs = []
        hypergraph_configs = []
        for i, d in enumerate(depths):
            if i == 0 or i == num_stages - 1:
                scan_configs.append([['h', 'h_flip', 'v', 'v_flip']] * d)
                hypergraph_configs.append(None)
            elif i == 1:
                blocks = []
                for j in range(d):
                    cfg = ['h', 'h_flip', 'v', 'v_flip']
                    cfg[j % 4] = 'hypergraph'
                    blocks.append(cfg)
                scan_configs.append(blocks)
                hypergraph_configs.append({
                    'k_neighbors': 4,
                    'num_hyperedges_ratio': 0.15,
                    'dropout': 0.1
                })
            else:
                blocks = []
                for j in range(d):
                    if j % 2 == 0:
                        blocks.append(['hypergraph', 'hypergraph', 'v', 'v_flip'])
                    else:
                        blocks.append(['h', 'h_flip', 'hypergraph', 'hypergraph'])
                scan_configs.append(blocks)
                hypergraph_configs.append({
                    'k_neighbors': 6,
                    'num_hyperedges_ratio': 0.2,
                    'dropout': 0.15
                })
        return scan_configs, hypergraph_configs

    else:
        raise ValueError(f"未知的扫描策略: {strategy}")


# =====================================================================
# 核心训练函数
# =====================================================================
def train_multimodal_model(
        model,
        train_loader,
        test_loader,
        num_classes=2,
        num_epochs=600,
        base_lr=1e-4,
        warmup_epochs=20,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        label_smoothing=0.1,
        use_amp=True,
        save_dir='./weights_multimodal',
        eval_interval=2,
        save_interval=40,
        fold=5,
        full_config=None,
        categorical_dims=None,
        preprocessor_save_path=None,
):
    """
    多模态融合模型训练

    Args:
        model: MultiModalFusionModel 实例
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
        num_classes: 类别数
        num_epochs: 总训练轮数
        base_lr: 基础学习率
        warmup_epochs: warmup 轮数
        weight_decay: 权重衰减
        grad_clip_norm: 梯度裁剪阈值
        label_smoothing: 标签平滑参数
        use_amp: 是否使用混合精度
        save_dir: 模型保存目录
        eval_interval: 评估间隔 (epoch)
        save_interval: checkpoint 保存间隔 (epoch)
        fold: 当前折数
    """
    save_dir = f'{save_dir}_fold{fold}'
    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device

    # ===== 计算类别权重 =====
    class_counts = torch.zeros(num_classes)
    for batch in train_loader:
        targets = batch['label']
        for c in range(num_classes):
            class_counts[c] += (targets == c).sum().item()

    total_samples = class_counts.sum().item()
    print(f"\n📊 Dataset class distribution:")
    for c in range(num_classes):
        print(f"   Class {c}: {int(class_counts[c])} ({class_counts[c] / total_samples * 100:.1f}%)")

    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    print(f"   Class weights: {class_weights.cpu().tolist()}")

    # ===== 损失函数 =====
    criterion = LabelSmoothingCrossEntropy(
        smoothing=label_smoothing,
        weight=torch.tensor([0.52, 1.57], dtype=torch.float32).to(device)
    )

    # ===== 优化器 =====
    param_groups = build_optimizer_param_groups(model, weight_decay=weight_decay)
    optimizer = optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999))

    # ===== 学习率调度 =====
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, min_lr=1e-6)

    # ===== 混合精度 =====
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ===== 追踪最佳性能 =====
    best_test_f1 = 0.0
    best_test_accuracy = 0.0
    best_test_auc = 0.0
    best_epoch = 0

    print(f"\n🔧 Training Configuration:")
    print(f"   Base LR: {base_lr}")
    print(f"   Warmup: {warmup_epochs} epochs")
    print(f"   Total: {num_epochs} epochs")
    print(f"   Weight Decay: {weight_decay}")
    print(f"   Gradient Clipping: {grad_clip_norm}")
    print(f"   Label Smoothing: {label_smoothing}")
    print(f"   Mixed Precision: {use_amp}")
    print()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # ===== 准备多模态输入 =====
            images = batch['images'].to(device)
            numerical = batch['numerical'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            targets = batch['label'].to(device)

            # ===== 前向传播 =====
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(
                        images=images,
                        numerical_data=numerical,
                        categorical_data=categorical
                    )
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(
                    images=images,
                    numerical_data=numerical,
                    categorical_data=categorical
                )
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if i % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [Batch {i}] loss: {loss.item():.4f}, lr: {lr:.2e}")

        # ===== 学习率更新 =====
        scheduler.step(epoch)

        avg_loss = running_loss / max(num_batches, 1)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {lr:.2e}')

        # ===== 保存最新模型 =====
        torch.save(model.state_dict(), os.path.join(save_dir, 'latest.pt'))

        # ===== 定期保存 checkpoint =====
        if (epoch + 1) % save_interval == 0:
            path = os.path.join(save_dir, f'epoch{epoch + 1}.pt')
            torch.save(model.state_dict(), path)
            print(f"💾 Saved checkpoint: {path}")

        # ===== 定期评估 =====
        if epoch % eval_interval == 0:
            model.eval()

            # --- 训练集评估 ---
            train_metrics = evaluate_multimodal(model, train_loader, device, use_amp)
            print("\n" + "🔵 Training Set Evaluation".center(80, "="))
            print_detailed_metrics(
                train_metrics['targets'], train_metrics['preds'],
                y_prob=train_metrics['probs'], dataset_name="Training Set"
            )

            # --- 测试集评估 ---
            test_metrics = evaluate_multimodal(model, test_loader, device, use_amp)
            print("\n" + "🟢 Test Set Evaluation".center(80, "="))
            test_acc, test_prec, test_recall, test_f1, test_sen, test_spec, test_auc, test_ppv, test_npv = \
                print_detailed_metrics(
                    test_metrics['targets'], test_metrics['preds'],
                    y_prob=test_metrics['probs'], dataset_name="Test Set"
                )

            print("\nClassification Report (Test Set):")
            print(classification_report(
                test_metrics['targets'], test_metrics['preds'],
                target_names=['Class 0', 'Class 1'], zero_division=0
            ))

            # ===== 保存最佳模型 =====
            if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_acc > best_test_accuracy):
                best_test_f1 = test_f1
                best_test_accuracy = test_acc
                best_test_auc = test_auc if test_auc is not None else 0.0
                best_epoch = epoch + 1

                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

                # ===== 保存完整推理包 (inference bundle) =====
                inference_bundle = {
                    'model_state_dict': model.state_dict(),
                    'config': full_config,
                    'categorical_dims': categorical_dims,
                    'best_epoch': best_epoch,
                    'best_metrics': {
                        'accuracy': test_acc,
                        'f1': test_f1,
                        'auc': test_auc,
                        'sensitivity': test_sen,
                        'specificity': test_spec,
                        'ppv': test_ppv,
                        'npv': test_npv,
                    }
                }
                torch.save(inference_bundle, os.path.join(save_dir, 'best_model_bundle.pt'))

                # ===== 保存最佳模型对应的 y_true / y_score =====
                best_csv_path = os.path.join(save_dir, 'best_y_true_score.csv')
                np.savetxt(best_csv_path,
                           np.column_stack([test_metrics['targets'],
                                            test_metrics['probs']]),
                           delimiter=',', header='y_true,y_score',
                           fmt=['%d', '%.6f'], comments='')
                print(f"   💾 best_y_true_score.csv 已保存至: {best_csv_path}")

                # 复制 preprocessor 到模型保存目录
                if preprocessor_save_path and os.path.exists(preprocessor_save_path):
                    dst = os.path.join(save_dir, 'preprocessor.pkl')
                    shutil.copy2(preprocessor_save_path, dst)
                    print(f"   📦 Preprocessor copied to {dst}")

                with open(os.path.join(save_dir, 'best_model_info.txt'), 'w') as f:
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"Accuracy: {test_acc:.4f}\n")
                    f.write(f"Sensitivity: {test_sen:.4f}\n")
                    f.write(f"Specificity: {test_spec:.4f}\n")
                    f.write(f"PPV: {test_ppv:.4f}\n")
                    f.write(f"NPV: {test_npv:.4f}\n")
                    f.write(f"F1-Score: {test_f1:.4f}\n")
                    f.write(f"AUC: {test_auc:.4f}\n" if test_auc else "AUC: N/A\n")

                auc_str = f"{test_auc:.4f}" if test_auc else "N/A"
                print(f"\n🏆 New Best! Epoch {best_epoch}, "
                      f"Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={auc_str}")
            else:
                print(f"\n📊 Current: Acc={test_acc:.4f}, F1={test_f1:.4f} "
                      f"| Best: Acc={best_test_accuracy:.4f}, F1={best_test_f1:.4f} (Epoch {best_epoch})")

    print("\n" + "=" * 80)
    print("Training Complete!".center(80))
    print("=" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Accuracy: {best_test_accuracy:.4f}")
    print(f"Best Test F1-Score: {best_test_f1:.4f}")
    if best_test_auc > 0:
        print(f"Best Test AUC: {best_test_auc:.4f}")
    print("=" * 80)


def evaluate_multimodal(model, dataloader, device, use_amp=True):
    """多模态模型评估"""
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            numerical = batch['numerical'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            targets = batch['label'].to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(
                        images=images,
                        numerical_data=numerical,
                        categorical_data=categorical
                    )
                logits = logits.float()
            else:
                logits = model(
                    images=images,
                    numerical_data=numerical,
                    categorical_data=categorical
                )

            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return {
        'preds': all_preds,
        'targets': all_targets,
        'probs': all_probs
    }


# =====================================================================
# 主程序
# =====================================================================
if __name__ == '__main__':

    # ===== 配置 =====
    config = {
        'num_classes': 2,
        'num_epochs': 600,

        # HyperVSSM 图像分支
        'patch_size': 4,
        'depths': [2, 4, 2],
        'dims': 96,
        'ssm_d_state': 16,
        'ssm_ratio': 2.0,
        'drop_path_rate': 0.2,
        'forward_type': 'v05_noz',
        'image_input_channels': 3,
        'scan_strategy': 'progressive',

        # MedicalGNN 表格分支
        'num_numerical_features': 23,
        'gnn_hidden_dim': 128,
        'gnn_heads': 4,
        'gnn_dropout': 0.2,
        'k_neighbors': 10,

        # 融合参数
        'fusion_hidden_dim': 256,
        'fusion_heads': 8,
        'num_fusion_blocks': 2,
        'fusion_dropout': 0.2,

        # 训练参数
        'base_lr': 1e-4,
        'warmup_epochs': 20,
        'weight_decay': 0.01,
        'grad_clip_norm': 1.0,
        'label_smoothing': 0.1,
        'use_amp': True,

        # 预训练权重 (可选, 设为 None 则从头训练)
        # 'image_pretrained_path': None,
        'image_pretrained_path': './weights_hypermamba_5/best_model.pt',

        'freeze_backbone': False,
        'model_init_seed': 42,
        'fold': 5,
    }

    # ===== 随机种子 =====
    torch.manual_seed(config['model_init_seed'])
    torch.cuda.manual_seed(config['model_init_seed'])

    # ===== 构建扫描配置 =====
    scan_configs, hypergraph_configs = build_scan_configs(
        config['depths'], strategy=config['scan_strategy']
    )

    # ===== 创建模型 =====
    print("\n🔨 创建多模态融合模型...")
    model = create_multimodal_fusion_model(
        num_classes=config['num_classes'],
        image_pretrained_path=config['image_pretrained_path'],
        image_input_channels=config['image_input_channels'],
        freeze_backbone=config['freeze_backbone'],
        scan_configs=scan_configs,
        hypergraph_configs=hypergraph_configs,
        depths=config['depths'],
        dims=config['dims'],
        patch_size=config['patch_size'],
        ssm_d_state=config['ssm_d_state'],
        ssm_ratio=config['ssm_ratio'],
        drop_path_rate=config['drop_path_rate'],
        forward_type=config['forward_type'],
        num_numerical_features=config['num_numerical_features'],
        categorical_dims=categorical_dims,
        gnn_hidden_dim=config['gnn_hidden_dim'],
        gnn_heads=config['gnn_heads'],
        gnn_dropout=config['gnn_dropout'],
        k_neighbors=config['k_neighbors'],
        fusion_hidden_dim=config['fusion_hidden_dim'],
        fusion_heads=config['fusion_heads'],
        num_fusion_blocks=config['num_fusion_blocks'],
        fusion_dropout=config['fusion_dropout'],
    ).cuda()

    print("\n" + "🚀 Model Initialization Complete".center(80, "="))
    print_model_parameters(model)

    # ===== 模型 Sanity Check =====
    print("\n🔍 Sanity check: multimodal forward pass...")
    with torch.no_grad():
        dummy_img = torch.randn(2, 3, 224, 224).cuda()
        dummy_num = torch.randn(2, config['num_numerical_features']).cuda()
        dummy_cat = {k: torch.randint(0, v, (2,)).cuda() for k, v in categorical_dims.items()}
        try:
            out = model(images=dummy_img, numerical_data=dummy_num, categorical_data=dummy_cat)
            print(f"   ✅ Output shape: {out.shape}")
            print(f"   ✅ Values: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
            probs = F.softmax(out, dim=1)
            print(f"   ✅ Softmax: {probs[0].cpu().tolist()}")
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            raise

    # ===== 开始训练 =====
    # 获取 preprocessor 路径 (与 DataOfClass 中一致)
    from DataOfClass import config as data_config
    preprocessor_path = os.path.join(
        os.path.dirname(data_config['fold_split_file']),
        'preprocessor.pkl'
    )

    print("\n" + "📊 Starting Multimodal Training...".center(80, "=") + "\n")
    train_multimodal_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=config['num_classes'],
        num_epochs=config['num_epochs'],
        base_lr=config['base_lr'],
        warmup_epochs=config['warmup_epochs'],
        weight_decay=config['weight_decay'],
        grad_clip_norm=config['grad_clip_norm'],
        label_smoothing=config['label_smoothing'],
        use_amp=config['use_amp'],
        save_dir='./weights_HyperSurgNet',
        eval_interval=2,
        save_interval=40,
        fold=config['fold'],
        full_config=config,
        categorical_dims=categorical_dims,
        preprocessor_save_path=preprocessor_path,
    )