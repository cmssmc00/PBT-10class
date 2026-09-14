# Pathology CREDIT：206 例、10 类、CONCH + 五项 IHC

本目录是本次实验的自包含运行目录。它只使用患者级 JSON、211 个已经由 CONCH
foundation model 处理的 HDF5 特征文件，以及 CREDIT 训练/评估代码；原始 WSI、原始
Excel、旧 181 例队列、旧四分类结果和 CIFAR 代码均不参与运行。

## 1. 队列与标签

- 206 位患者，211 张 WSI；同一患者的多张 WSI 始终合并为一个患者级 patch bag。
- 每个 H5 的 `features` 形状为 `[patch 数, 768]`，共 1,669,413 个 patch。
- 每位患者都有 `GFAP / Synaptophysin / INI1 / H3K27M / ALK1` 五项 POS/NEG。
- 原 Excel 中正体与斜体（inferred）值全部作为有效值使用；斜体来源仍保存在
  `ihc_inferred` 中，仅用于溯源，不作为额外模型特征。
- 配置固定为 `ihc_policy: all`，所以进入模型的 `ihc_mask` 对每例都是
  `[1, 1, 1, 1, 1]`，不会再把斜体值乘成 0。
- 保留全部 206 例，包括 5 例 `Not tumor`，以及 1 例分类为
  `Non-glial tumor` 但源数据 `tumor_flag=0` 的病例；配置因此保持
  `exclude_tumor_flag_zero: false`。

类别顺序和患者数固定如下：

1. Glial tumors, low-grade — 69
2. Glial tumors, high-grade — 23
3. Glio-neuronal tumors — 30
4. Embryonal tumors, high-grade — 23
5. Ependymal tumors — 13
6. Uncertain histiogenesis tumors (these are defined by genetics or methylation classs) — 14
7. Choroid plexus tumor — 4
8. Pineal tumors — 7
9. Non-glial tumor — 18
10. Not tumor — 5

第 6 类名称中的 `classs` 是源数据标签的原始拼写，代码按原值保留，避免标签不匹配。

## 2. 模型数据流

```text
一个患者的全部 CONCH patch: [P, 768]
        │
        └── Dense(256) + Gated Attention MIL ──► 病理向量 [256]

IHC 值 [5] × ihc_mask [5]（本队列 mask 全为 1）
        │
        └── Dense(32, GELU) + LayerNorm + Dense(32, GELU)
                                                └── IHC embedding [32]

病理向量 [256] + IHC embedding [32]
        └── concatenate + fusion Dense(128) ──► 融合向量 [128]
                                                ├── Teacher logits [10]
                                                └── CREDIT output [21]
```

`P` 是一位患者合并其所有 WSI 后的 patch 总数；`batch` 是一次同时送入模型的患者数。
变长 patch bag 在 batch 内会 padding，并由 `patch_mask` 排除补齐位置。

CREDIT student 的 21 维来自 `2C+1`，其中 `C=10`：前 10 维生成交点概率，接着
10 维生成各类别概率区间长度，最后 1 维生成 `beta`。五个 teacher 结构和超参数
相同，但使用五个独立随机种子训练；student 只使用冻结 ensemble 产生的 credal target。

## 3. 患者级五折

`splits/pathology_5fold.json` 是固定患者级 5 折：每折约 70% train、10% eval、20%
test；五个 test 互不重叠并恰好覆盖 206 位患者一次。同一患者的多张 WSI 不会跨分区。

`Choroid plexus tumor` 只有 4 例，所以在 5 折中必然有一个 test fold 没有该类；这是
数据规模决定的正常现象。每个训练折仍包含全部 10 类。正式结论应使用五折合并后的
206 例 OOF 结果，不应只报告单折。

## 4. 环境与检查

```bash
cd /Users/mingshi/Desktop/pbt-new/credit
source .venv-pathology/bin/activate
```

重新验证所有 H5 的形状、类型和 NaN/Inf：

```bash
python scripts/validate_pathology_data.py \
  --config configs/pathology_credit.yaml \
  --check-values
```

如需按固定 seed 重新生成同一 5 折：

```bash
python scripts/make_patient_splits.py \
  --config configs/pathology_credit.yaml
```

运行两例患者的 teacher/student 前向、反向与临时 checkpoint 测试：

```bash
python scripts/smoke_test_pathology.py \
  --config configs/pathology_credit.yaml \
  --fold 0
```

## 5. 正式训练与评估

每一折依次运行：

```bash
python scripts/train_teachers.py --config configs/pathology_credit.yaml --fold 0
python scripts/train_credit.py   --config configs/pathology_credit.yaml --fold 0
python scripts/evaluate_credit.py --config configs/pathology_credit.yaml --fold 0
```

其余四折：

```bash
for fold in 1 2 3 4; do
  python scripts/train_teachers.py --config configs/pathology_credit.yaml --fold "$fold"
  python scripts/train_credit.py --config configs/pathology_credit.yaml --fold "$fold"
  python scripts/evaluate_credit.py --config configs/pathology_credit.yaml --fold "$fold"
done
```

全部五折评估结束后汇总 OOF：

```bash
python scripts/aggregate_oof.py \
  --config configs/pathology_credit.yaml \
  --bootstrap-samples 2000
```

脚本默认拒绝覆盖 checkpoint；确定重训时才向训练命令加入 `--overwrite`。本次只准备并
验证代码与数据，没有启动正式训练。

## 6. 关键文件

```text
configs/pathology_credit.yaml        固定实验配置
data/patients_206.json               206 例患者标签、IHC 与 WSI 映射
data/cohort_summary.json             队列筛选口径与统计
data/conch_features/*.h5             211 个 CONCH patch 特征
splits/pathology_5fold.json          固定患者级五折
pathology_credit/                    数据、模型、CREDIT loss、训练与指标实现
scripts/                             验证、训练、评估和 OOF 汇总入口
```

默认输出目录是
`outputs/pathology_credit/conch_ihc_credit_206_10class/`，首次训练时由脚本自动创建。

## 7. 斜体 IHC 强制为 NEG 的配对实验

当前主实验及其结果继续使用 `configs/pathology_credit.yaml`，其中正体和斜体 IHC
全部按原 POS/NEG 使用。第二份配置
`configs/pathology_credit_inferred_neg.yaml` 保持206例患者、H5、五折划分、模型结构、
随机种子和训练超参数完全相同，只把每个斜体/inferred IHC 项置为 `0=NEG`。

该配置使用 `ihc_policy: mask_inferred` 和 `ihc_mask_as_feature: false`。因此斜体项经过
`IHC × mask` 后为0，但模型不会把 mask 本身作为分类输入。正体的 POS/NEG 保持不变。
新实验输出写入独立目录：

```text
outputs/pathology_credit/conch_ihc_credit_206_10class_inferred_as_neg/
```

先进行轻量检查：

```bash
./.venv-pathology/bin/python scripts/validate_pathology_data.py \
  --config configs/pathology_credit_inferred_neg.yaml \
  --check-values

./.venv-pathology/bin/python scripts/smoke_test_pathology.py \
  --config configs/pathology_credit_inferred_neg.yaml \
  --fold 0
```

随后运行完整五折：

```bash
caffeinate -i zsh -c '
set -e
python_cmd="./.venv-pathology/bin/python"

for fold in 0 1 2 3 4; do
  "$python_cmd" scripts/train_teachers.py \
    --config configs/pathology_credit_inferred_neg.yaml \
    --fold "$fold"

  "$python_cmd" scripts/train_credit.py \
    --config configs/pathology_credit_inferred_neg.yaml \
    --fold "$fold"

  "$python_cmd" scripts/evaluate_credit.py \
    --config configs/pathology_credit_inferred_neg.yaml \
    --fold "$fold"
done

"$python_cmd" scripts/aggregate_oof.py \
  --config configs/pathology_credit_inferred_neg.yaml \
  --bootstrap-samples 2000
'
```

不要给这组命令添加 `--overwrite`；两个实验名称不同，不会读取或覆盖当前 all-IHC
实验的 checkpoint 和 OOF 汇总。

## 8. CREDIT loss 尺度

当前 `pathology_credit/ced.py` 保持论文式 CED：分类交叉熵、10 个类别区间长度的平方误差
之和，以及一维 `beta` 平方误差，总体再乘蒸馏温度平方。由于区间误差沿类别维使用
sum，当前 10 类版本在相同逐类误差下约为旧仓库 mean 写法的 10 倍；这只是 loss
定义尺度差异，不改变输出维度含义。

## 9. WSI + 年龄/部位先验 + 病理层级实验

`configs/pathology_credit_wsi_hierarchical_prior.yaml` 是新增且完全独立的实验；它不会
读取或覆盖前面的 IHC 实验结果。Teacher 和 Student 均不接收五项 IHC，只接收 CONCH
patch bag、年龄和标准化后的肿瘤位置。运行时临床信息来自
`data/clinical_206.json`：206/206 例有年龄，196/206 例有明确位置，其余 10 例使用
`Unknown` 位置并令 `location_mask=0`。这个 JSON 可由
`scripts/build_clinical_metadata.py` 从 Excel 逐行核对后重建；位置规则只匹配解剖词，
不读取类别标签。

模型共用一个 Gated-Attention WSI encoder，然后按病理学家的顺序产生三个条件输出：

```text
stage 1: C10 vs C1-C9          -> p10
stage 2: C9 vs C1-C8           -> p9_given_not_C10
stage 3: C1-C8                 -> q1 ... q8

P(C10) = p10
P(C9)  = (1-p10) * p9_given_not_C10
P(Ci)  = (1-p10) * (1-p9_given_not_C10) * qi, i=1...8
```

年龄先除以固定的 25 年尺度后进入 8 维 MLP，位置进入 8 维 embedding；二者与各自的
availability mask 合并成 16 维临床向量。三个节点分别计算临床 prior logit，再通过
限制在 0 到 1 的可学习系数加入 WSI logit。三个系数初始均为 0.1，避免小样本临床
先验在训练开始时压过 WSI，之后仅由当前 fold 的 train 病例更新。最终十类 log
probability 仍占 CREDIT 输出的前 10 维，后 11 维仍是 interval length 和 beta，所以
原评估与 OOF 汇总格式保持兼容。评估结果还会记录三级准确率和三个最终 prior 系数。

先做检查：

```bash
./.venv-pathology/bin/python scripts/validate_pathology_data.py \
  --config configs/pathology_credit_wsi_hierarchical_prior.yaml \
  --check-values

./.venv-pathology/bin/python scripts/smoke_test_pathology.py \
  --config configs/pathology_credit_wsi_hierarchical_prior.yaml \
  --fold 0
```

正式五折运行时，把第 5 节命令里的配置统一替换为
`configs/pathology_credit_wsi_hierarchical_prior.yaml`。输出写入独立目录：

```text
outputs/pathology_credit/conch_wsi_hierarchical_clinical_prior_credit_206_10class/
```

当前只完成数据验证和两例 smoke test，未启动正式五折训练。

## 10. 普通十分类 WSI + 年龄/位置 prior

`configs/pathology_credit_wsi_clinical_prior.yaml` 是不使用层级结构的配对实验。WSI
embedding 直接生成 10 个 logits，年龄/位置临床分支也生成 10 个 prior logits，然后按
`final_logits = wsi_logits + gamma * prior_logits` 融合。`gamma` 限制在 0 到 1，初始值为
0.1；IHC 完全不进入数据 batch 或模型。该实验写入独立目录：

```text
outputs/pathology_credit/conch_wsi_clinical_prior_credit_206_10class/
```

其运行方法与第 5 节相同，只需把配置替换为
`configs/pathology_credit_wsi_clinical_prior.yaml`。

## 11. WSI + 层级、无临床 prior

`configs/pathology_credit_wsi_hierarchical.yaml` 仅启用病理层级
`C10 -> C9 -> C1-C8`，不使用年龄、位置或 IHC。它是用来单独测量层级结构贡献的消融
实验，输出写入：

```text
outputs/pathology_credit/conch_wsi_hierarchical_credit_206_10class/
```
