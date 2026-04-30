import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
import pandas as pd
from torchvision.transforms import transforms
import json
import random
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ================================================================
# 全局随机种子（只设一次）
# ================================================================
random.seed(42)
np.random.seed(42)

# ================================================================
# ★ 用户配置区：分类型特征列索引（在特征矩阵中的 0-based 列号）
#   例如：第2列和第5列是分类型，则写 [2, 5]
#   数值型列 = 所有列 - 分类型列，自动推断，无需手动列举
# ================================================================
CATEGORICAL_COLS = []          # ← 根据实际数据修改，例如 [2, 5, 7]
NUM_FEATURE_COLS = 37          # 特征总列数（Excel 第1~18列）

# 所有列索引
_all_cols        = list(range(NUM_FEATURE_COLS))
_numerical_cols  = [c for c in _all_cols if c not in CATEGORICAL_COLS]

# ================================================================
# ★ 图像视角关键词配置
#   程序会在文件名（小写）中逐一查找以下关键词来判断视角
#   如需新增别名，直接在对应列表里添加即可
# ================================================================
VIEW_KEYWORDS = {
    '立位': ['立位'],
    '侧位': ['侧位'],
    '卧位': ['卧位'],
}
# 通道顺序：立位→通道0，侧位→通道1，卧位→通道2
CHANNEL_ORDER = ['立位', '侧位', '卧位']


# ================================================================
# 1. 读取 Excel
# ================================================================
ReadFile = pd.read_csv('/root/autodl-tmp/SAViT/data/clinical_data.csv', dtype={0: str})
print("=" * 60)
print("Excel文件信息:")
print(f"  总行数: {len(ReadFile)}")
print(f"  总列数: {len(ReadFile.columns)}")

TupianGroupIDs = ReadFile.iloc[:, 0].astype(str).str.strip().tolist()
Labels         = np.array(ReadFile.iloc[:, 1 + NUM_FEATURE_COLS], dtype=np.int64)

# ── 读取原始特征块（保留 object 类型，避免字符串列强转 float 报错）──
_raw_df = ReadFile.iloc[:, 1:1 + NUM_FEATURE_COLS].copy()

# ── 自动发现分类型列（含非数值字符串的列）──
_auto_cat_cols = []
for col_idx in range(NUM_FEATURE_COLS):
    col = _raw_df.iloc[:, col_idx]
    # 排除全为 NaN 的列
    non_null = col.dropna()
    if len(non_null) == 0:
        continue
    # 尝试转 float；如果有任意一个值转不了，就是分类型列
    try:
        pd.to_numeric(non_null, errors='raise')
    except (ValueError, TypeError):
        _auto_cat_cols.append(col_idx)

# 用户手动配置的列 + 自动发现的列合并（去重）
CATEGORICAL_COLS = sorted(set(CATEGORICAL_COLS) | set(_auto_cat_cols))
_numerical_cols  = [c for c in _all_cols if c not in CATEGORICAL_COLS]

# ── 对分类型列做 LabelEncoding（字符串 → 整数，NaN 保留为 NaN）──
# 编码映射保存到全局，供后续调试或推理复用
CAT_LABEL_ENCODERS: dict[int, dict] = {}   # col_idx → {原始值: 整数编码}

for col_idx in CATEGORICAL_COLS:
    col     = _raw_df.iloc[:, col_idx]
    uniques = sorted(col.dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(uniques)}
    CAT_LABEL_ENCODERS[col_idx] = mapping
    _raw_df.iloc[:, col_idx] = col.map(
        lambda x, m=mapping: m.get(str(x), np.nan) if pd.notna(x) else np.nan
    )
    print(f"  分类列[{col_idx}] 编码映射: {mapping}")

# ── 转为 float64 矩阵（此时所有列均为数值或 NaN）──
Features_raw = _raw_df.astype(np.float64).values   # (N, 18)

print(f"  特征矩阵形状: {Features_raw.shape}")
print(f"  缺失值数量:   {np.isnan(Features_raw).sum()}")
print(f"  数值型列({len(_numerical_cols)}个): {_numerical_cols}")
print(f"  分类型列({len(CATEGORICAL_COLS)}个): {CATEGORICAL_COLS}")
print(f"  标签分布:     0={np.sum(Labels==0)}, 1={np.sum(Labels==1)}")
print("=" * 60)

# ================================================================
# 2. 创建 ID → 索引的映射（匹配带/不带后缀的ID）
# ================================================================
Dict_tupianzu_to_label   = {}   # id -> int label
Dict_tupianzu_to_index   = {}   # id -> row index in Features_raw
Dict_tupianzu_to_feature = {}   # id -> np.array 特征（每折预处理后更新）

for i, full_id in enumerate(TupianGroupIDs):
    Dict_tupianzu_to_label[full_id]  = int(Labels[i])
    Dict_tupianzu_to_index[full_id]  = i

    if '-' in full_id:
        base_id = full_id.split('-')[0]
        if base_id not in Dict_tupianzu_to_label:
            Dict_tupianzu_to_label[base_id]  = int(Labels[i])
            Dict_tupianzu_to_index[base_id]  = i

print(f"标签字典大小: {len(Dict_tupianzu_to_label)}")
print("=" * 60)


# ================================================================
# 3. 图像变换
# ================================================================
TransTrain = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

TransTest = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ================================================================
# 4. 辅助函数：视角识别
# ================================================================
def detect_view_type(filename: str) -> str | None:
    """
    根据文件名（不区分大小写）识别视角类型。
    返回 '立位' | '侧位' | '卧位' | None（无法识别时）
    """
    lower = filename.lower()
    for view, keywords in VIEW_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower:
                return view
    return None


def load_view_images(img_dir: str, transform) -> dict:
    """
    扫描 img_dir 下所有图像文件，按视角分类。
    返回: {'立位': [PIL.Image, ...], '侧位': [...], '卧位': [...]}
    每种视角可能有 0 个或多个图像（取第一个）。
    """
    view_images: dict[str, list] = {v: [] for v in CHANNEL_ORDER}
    unrecognized = []

    try:
        files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.PNG', '.bmp', '.tif', '.tiff'))]
    except Exception as e:
        print(f"  ⚠️  无法列举目录 {img_dir}: {e}")
        return view_images

    for fname in files:
        view = detect_view_type(fname)
        if view and view in view_images:
            view_images[view].append(os.path.join(img_dir, fname))
        else:
            unrecognized.append(fname)

    if unrecognized:
        # 尝试按文件名排序兜底分配（仅在全部无法识别时）
        if all(len(v) == 0 for v in view_images.values()) and len(files) >= 1:
            sorted_files = sorted(files)
            for idx, view in enumerate(CHANNEL_ORDER):
                if idx < len(sorted_files):
                    view_images[view].append(os.path.join(img_dir, sorted_files[idx]))

    return view_images


def build_three_channel_image(img_dir: str, transform) -> torch.Tensor:
    """
    构建 3 通道图像张量，通道顺序 = [立位, 侧位, 卧位]。
    缺失视角（侧位/卧位）统一用立位图像填充。
    若连立位也缺失，返回全零张量并打印警告。

    返回: (3, H, W) 的 FloatTensor
    """
    view_images = load_view_images(img_dir, transform)

    # 先确定立位图像（后续填充用）
    liwei_path = view_images['立位'][0] if view_images['立位'] else None

    channels = []
    for view in CHANNEL_ORDER:          # 立位 → 侧位 → 卧位
        path_list = view_images[view]

        if path_list:
            img_path = path_list[0]     # 同一视角有多张时取第一张
        elif liwei_path is not None:
            # 缺失当前视角，用立位填充
            img_path = liwei_path
        else:
            # 立位也没有：生成全零通道
            channels.append(torch.zeros(1, 448, 448))
            continue

        try:
            img = Image.open(img_path).convert('L')
            channels.append(transform(img))   # (1, H, W)
        except Exception as e:
            print(f"  ❌ 读取图片失败 [{view}]: {img_path}  原因: {e}")
            if liwei_path and img_path != liwei_path:
                # 读取失败时也尝试立位
                try:
                    img = Image.open(liwei_path).convert('L')
                    channels.append(transform(img))
                    continue
                except Exception:
                    pass
            channels.append(torch.zeros(1, 448, 448))

    # 视角识别统计（调试用，可注释掉）
    missing = [v for v in CHANNEL_ORDER if not view_images[v]]
    if missing:
        pass   # 打印太多会影响训练速度；需调试时取消注释：
        # print(f"  ℹ️  {os.path.basename(img_dir)}: 缺失视角 {missing}，已用立位填充")

    return torch.cat(channels, dim=0)   # (3, H, W)


# ================================================================
# 5. ID 匹配函数
# ================================================================
def find_matching_id(folder_name: str, dict_keys: set):
    """尝试多种策略将文件夹名匹配到 Excel ID"""
    if folder_name in dict_keys:
        return folder_name
    if '-' in folder_name:
        base = folder_name.split('-')[0]
        if base in dict_keys:
            return base
    for key in dict_keys:
        base_key = key.split('-')[0] if '-' in key else key
        if base_key == folder_name:
            return key
    folder_digits = ''.join(filter(str.isdigit, folder_name))
    for key in dict_keys:
        if ''.join(filter(str.isdigit, key)) == folder_digits:
            return key
    return None


# ================================================================
# 6. 扫描磁盘（只扫描一次，所有折共用）
#    ★ 放宽条件：允许图像不足3张，缺失视角在 __getitem__ 中用立位填充
# ================================================================
def scan_all_samples(root: str) -> dict:
    """
    扫描数据根目录下 baoshou / shoushu 两个子文件夹。
    返回: {matched_id: (img_dir, folder_name)}
    ★ 只要目录非空（≥1张图）就纳入，不再要求恰好3张。
    """
    all_samples = {}
    unmatched   = []
    dict_keys   = set(Dict_tupianzu_to_label.keys())

    img_exts = {'.jpg', '.jpeg', '.PNG', '.bmp', '.tif', '.tiff'}

    for subfolder in ['baoshou', 'shoushu']:
        subfolder_path = os.path.join(root, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"⚠️  警告: 子文件夹不存在 → {subfolder_path}")
            continue

        for item in os.listdir(subfolder_path):
            item_path = os.path.join(subfolder_path, item)
            if not os.path.isdir(item_path):
                continue

            # 统计图像文件数量（至少1张才纳入）
            img_count = sum(
                1 for f in os.listdir(item_path)
                if os.path.splitext(f)[1].lower() in img_exts
            )
            if img_count < 1:
                continue

            matched_id = find_matching_id(item, dict_keys)
            if matched_id:
                # 统计各视角数量（供诊断）
                files = os.listdir(item_path)
                view_counts = {v: sum(1 for f in files if detect_view_type(f) == v)
                               for v in CHANNEL_ORDER}
                missing_views = [v for v, cnt in view_counts.items() if cnt == 0]
                if missing_views:
                    pass  # 调试时取消注释：
                    # print(f"  ℹ️  {subfolder}/{item}: 缺失视角 {missing_views}")
                all_samples[matched_id] = (item_path, item)
            else:
                unmatched.append(f"{subfolder}/{item}")

    print(f"✅ 磁盘扫描完成: {len(all_samples)} 个有效样本, "
          f"{len(unmatched)} 个未匹配")
    if unmatched:
        print(f"   未匹配示例（前5）: {unmatched[:5]}")
    return all_samples


# ================================================================
# 7. 表格特征预处理
#    ★ 数值型列：均值填充 + StandardScaler
#    ★ 分类型列：众数填充，不做标准化（保持原始整数编码）
#    ⚠️ 只用训练集 fit，再 transform 全部数据，防止数据泄露
# ================================================================
def preprocess_features(train_ids: list, save_path: str = None):
    """
    对特征矩阵做缺失值填充和标准化，结果写回 Dict_tupianzu_to_feature。

    分列策略：
      - 数值型列（_numerical_cols）：均值填充 → StandardScaler 标准化
      - 分类型列（CATEGORICAL_COLS）：众数填充 → 保持原值（不标准化）

    参数:
        train_ids  : 当前折训练集 ID 列表
        save_path  : 可选，保存预处理器的路径（.pkl）
    """
    global Dict_tupianzu_to_feature

    # ── 找训练集行索引 ──
    train_idx_set = set()
    for tid in train_ids:
        if tid in Dict_tupianzu_to_index:
            train_idx_set.add(Dict_tupianzu_to_index[tid])
        elif '-' in tid:
            base = tid.split('-')[0]
            if base in Dict_tupianzu_to_index:
                train_idx_set.add(Dict_tupianzu_to_index[base])

    train_row_indices = sorted(train_idx_set)
    train_features    = Features_raw[train_row_indices]

    print(f"   预处理: 训练集行数={len(train_row_indices)}, "
          f"原始缺失值={np.isnan(train_features).sum()}")

    # ── 初始化输出矩阵（先复制原始数据，再逐列替换）──
    n_all = Features_raw.shape[0]
    Features_processed = Features_raw.copy()     # (N, 18) float64，仍含NaN

    # ── 数值型列：均值填充 + StandardScaler ──
    num_imputer = None
    num_scaler  = None
    if _numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X_num_train = train_features[:, _numerical_cols]
        num_imputer.fit(X_num_train)

        X_num_all = num_imputer.transform(Features_raw[:, _numerical_cols])

        num_scaler = StandardScaler()
        num_scaler.fit(X_num_all[train_row_indices])
        X_num_scaled = num_scaler.transform(X_num_all)

        for new_pos, col in enumerate(_numerical_cols):
            Features_processed[:, col] = X_num_scaled[:, new_pos]

        print(f"   数值型列标准化后: "
              f"均值≈{X_num_scaled[train_row_indices].mean():.4f}, "
              f"标准差≈{X_num_scaled[train_row_indices].std():.4f}")

    # ── 分类型列：众数填充，不标准化 ──
    cat_imputer = None
    if CATEGORICAL_COLS:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat_train = train_features[:, CATEGORICAL_COLS]
        cat_imputer.fit(X_cat_train)

        X_cat_all = cat_imputer.transform(Features_raw[:, CATEGORICAL_COLS])

        for new_pos, col in enumerate(CATEGORICAL_COLS):
            Features_processed[:, col] = X_cat_all[:, new_pos]

        # 分类型列缺失值统计
        cat_missing_before = np.isnan(train_features[:, CATEGORICAL_COLS]).sum()
        print(f"   分类型列(众数填充): 训练集填补了 {cat_missing_before} 个缺失值, "
              f"不做标准化")

    # ── 写回全局字典 ──
    for i, full_id in enumerate(TupianGroupIDs):
        Dict_tupianzu_to_feature[full_id] = Features_processed[i].astype(np.float32)
        if '-' in full_id:
            base_id = full_id.split('-')[0]
            if base_id not in Dict_tupianzu_to_feature:
                Dict_tupianzu_to_feature[base_id] = Features_processed[i].astype(np.float32)

    # ── 保存预处理器 ──
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            'num_imputer':       num_imputer,
            'num_scaler':        num_scaler,
            'cat_imputer':       cat_imputer,
            'numerical_cols':    _numerical_cols,
            'categorical_cols':  CATEGORICAL_COLS,
            'cat_label_encoders': CAT_LABEL_ENCODERS,   # ★ 新增：字符串→整数映射
            'num_feature_cols':  NUM_FEATURE_COLS,
        }, save_path)
        print(f"   ✅ 预处理器已保存 → {save_path}")

    return num_imputer, num_scaler, cat_imputer


# ================================================================
# 8. Dataset 类
# ================================================================
class KFoldDataset(torch.utils.data.Dataset):
    """
    基于预定义 JSON 折划分的数据集，支持图像（立位/侧位/卧位）+ 表格多模态输入。
    缺失视角自动用立位图像填充。
    """
    def __init__(self, all_samples: dict, id_list: list,
                 is_train: bool = True, split_name: str = 'train'):
        self.is_train       = is_train
        self.PathsOfDirs    = []
        self.tupianzu_names = []
        self.folder_names   = []
        missing_in_disk     = []

        for pid in id_list:
            if pid in all_samples:
                img_dir, folder_name = all_samples[pid]
                lookup_id = pid
            else:
                base = pid.split('-')[0] if '-' in pid else pid
                if base in all_samples:
                    img_dir, folder_name = all_samples[base]
                    lookup_id = base if base in Dict_tupianzu_to_label else pid
                else:
                    missing_in_disk.append(pid)
                    continue

            if lookup_id not in Dict_tupianzu_to_label:
                missing_in_disk.append(pid)
                continue

            self.PathsOfDirs.append(img_dir)
            self.tupianzu_names.append(lookup_id)
            self.folder_names.append(folder_name)

        labels_in_set = [Dict_tupianzu_to_label[n] for n in self.tupianzu_names]
        print(f"  [{split_name}] 样本数={len(self.PathsOfDirs)}, "
              f"标签0={labels_in_set.count(0)}, 标签1={labels_in_set.count(1)}")
        if missing_in_disk:
            print(f"  ⚠️  JSON 中 {len(missing_in_disk)} 个 ID 在磁盘上找不到: "
                  f"{missing_in_disk[:5]}")

    def __getitem__(self, index):
        matched_id = self.tupianzu_names[index]
        img_dir    = self.PathsOfDirs[index]
        transform  = TransTrain if self.is_train else TransTest

        # ── 读取三视角图像（立位/侧位/卧位），缺失视角用立位填充 ──
        try:
            img = build_three_channel_image(img_dir, transform)   # (3, H, W)
        except Exception as e:
            print(f"❌ 读取图片失败: {img_dir}  原因: {e}")
            img = torch.zeros(3, 448, 448)

        # ── 标签 ──
        label = Dict_tupianzu_to_label[matched_id]

        # ── 表格特征（已标准化/众数填充）──
        if matched_id in Dict_tupianzu_to_feature:
            features = torch.FloatTensor(Dict_tupianzu_to_feature[matched_id])
        else:
            print(f"⚠️  特征缺失: {matched_id}，用零向量填充")
            features = torch.zeros(NUM_FEATURE_COLS)

        return img, label, features

    def __len__(self):
        return len(self.PathsOfDirs)


# ================================================================
# 9. 核心接口：加载指定折的 train/test loader
# ================================================================
def create_kfold_loaders(
    root:           str,
    fold_split_dir: str,
    fold_index:     int,
    all_samples:    dict,
    batch_size:     int  = 6,
    num_workers:    int  = 4,
    preprocessor_save_dir: str = None,
):
    """
    加载第 fold_index 折的训练集和测试集 DataLoader。

    参数:
        root                  : 图像根目录
        fold_split_dir        : 存放 dataset_split_fold_*.json 的目录
        fold_index            : 1 ~ 5
        all_samples           : scan_all_samples() 的返回结果（复用）
        batch_size            : DataLoader batch 大小
        num_workers           : DataLoader 工作进程数
        preprocessor_save_dir : 预处理器保存目录（None 则不保存）

    返回:
        train_loader, test_loader
    """
    assert 1 <= fold_index <= 5, "fold_index 必须在 1~5 之间"

    json_path = os.path.join(fold_split_dir, f"dataset_split_fold_{fold_index}.json")
    assert os.path.exists(json_path), f"找不到划分文件: {json_path}"

    with open(json_path, 'r') as f:
        fold_data = json.load(f)

    train_ids = fold_data['split']['train']
    test_ids  = fold_data['split']['test']

    print(f"\n{'=' * 60}")
    print(f"📂 第 {fold_index} / {fold_data['config']['total_folds']} 折  "
          f"(seed={fold_data['config']['random_seed']})")
    print(f"   JSON: 训练={len(train_ids)}, 测试={len(test_ids)}")

    # ── 每折独立做特征预处理（防止数据泄露）──
    save_path = None
    if preprocessor_save_dir:
        save_path = os.path.join(preprocessor_save_dir,
                                 f"preprocessor_fold_{fold_index}.pkl")
    print("  ▶ 特征预处理中...")
    preprocess_features(train_ids, save_path=save_path)

    # ── 构建 Dataset ──
    train_dataset = KFoldDataset(all_samples, train_ids,
                                 is_train=True,  split_name='train')
    test_dataset  = KFoldDataset(all_samples, test_ids,
                                 is_train=False, split_name='test')

    # ── 验证无重叠 ──
    overlap = set(train_dataset.PathsOfDirs) & set(test_dataset.PathsOfDirs)
    if overlap:
        print(f"⚠️  警告: 训练集与测试集存在 {len(overlap)} 个重叠样本!")
    else:
        print("  ✅ 无重叠验证通过")

    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader


# ================================================================
# 10. 五折交叉验证主循环
# ================================================================
ROOT                  = r"/root/autodl-tmp/SAViT/data/"
FOLD_SPLIT_DIR        = r"./kfold_splits/"
PREPROCESSOR_SAVE_DIR = r"./kfold_splits/"
BATCH_SIZE            = 6
NUM_WORKERS           = 4

all_samples = scan_all_samples(ROOT)

train_loader, test_loader = create_kfold_loaders(
    root                  = ROOT,
    fold_split_dir        = FOLD_SPLIT_DIR,
    fold_index            = 1,
    all_samples           = all_samples,
    batch_size            = BATCH_SIZE,
    num_workers           = NUM_WORKERS,
    preprocessor_save_dir = PREPROCESSOR_SAVE_DIR,
)