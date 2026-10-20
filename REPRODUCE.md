# 复现说明（GNet / 背景引导视频前景分割）

本仓库为对原稿“背景引导视频前景分割”的可复现重建、泄漏审计与路线 C（无泄漏因果背景 + 可靠性门控）实验。

---

## 1. 环境

- Python 3.11.7；PyTorch 2.13.0（CUDA 13.0）；NVIDIA L20 ×2。
- 依赖见 `requirements.txt`：`pip install -r requirements.txt`
- 所有命令在 `gnet_seg/` 下执行，并 `export PYTHONPATH=.`

---

## 2. 数据

| 数据 | 用途 | 路径 | 来源/说明 |
|---|---|---|---|
| CDnet 2014（53 视频/11 类） | 主评测 | `datasets/cdnet/CDnet14/<cat>/<video>/{input,groundtruth,ROI.bmp,temporalROI.txt}` | changedetection.net；含 D2-CDNet 打包的 `bgs_subsense/`（SuBSENSE 掩膜）与 `backgrounds/`（静态背景=全段时序中值，**未来帧 oracle**） |
| Cityscapes（背景） | 合成训练 | `datasets/raw/cityscapes_train.parquet` | 街景无标注图 |
| COCO 2017 val（目标） | 合成训练 | `datasets/raw/val2017/` + `datasets/raw/annotations/instances_val2017.json` | 实例掩膜 |
| SBI 2015（14 序列） | 跨数据集 | `datasets/sbi_root/sbi/`（软链到 FgSegNet 自带副本） | 布局同 CDnet |

获取：
- CDnet/SBI 见 `scripts/fetch_data.sh`、`scripts/fetch_cdnet.sh`、`scripts/fetch_baselines.sh`。
- 数据准备/索引见 `scripts/prepare_data.sh`。

---

## 3. 冻结评测协议（`utils/protocol.py`, v1）

- temporalROI 内**全部帧**（`frame_stride=1`）。
- GT：前景=255；背景=0 或 50（shadow 记背景）；忽略=85（ROI 外）或 170（unknown）。
- `ROI.bmp` 外（<128）置为忽略。
- 指标：逐视频 F1/PWC → 类别平均（`AVG_category`）；同时报视频平均。
- 背景来源三级：`causal`（仅过去帧、无标签）、`leaky`（因果但含 SuBSENSE 决策=`reconstructed/running`）、`oracle`（未来帧=`provided/median`）。
- **可报告结果必须用 causal**；leaky 仅用于泄漏对照，oracle 仅诊断。程序内置守卫（`--allow-leaky/--allow-oracle`）。
- 每次评测输出 CSV + JSON sidecar（含协议、背景来源、逐类指标）。

---

## 4. 训练

### 4.1 主模型（合成监督）
```bash
python3 train.py --seed 0 --batch 8 --workers 6 --size 513 \
  --epochs 8 --steps-per-epoch 600 --lr 0.01 --out checkpoints/gnet_main_seed0
```
变体：`--no-dda / --no-pmb / --no-tsam / --tsam-depth {8,16} / --fusion {sub,cat} / --no-bg`。

### 4.2 M1 可靠性门控
```bash
python3 train.py --rel-gate --seed 0 ... --out checkpoints/gnet_m1
```
合成背景（PMB）同时给出可靠性监督（`data/synth.py: with_rel`）。

### 4.3 完整方法（M1+M3+T3）
```bash
python3 train.py --rel-gate --boundary --degrade --shadow --synth-bg causal \
  --seed 0 ... --out checkpoints/gnet_full
```
`--synth-bg causal` = L2b：合成端用与推理相同的因果 EMA 估计器生成训练背景与可靠性
（T3 退化强度见 `logs/t3_calibration.md`，已按真实 CDnet 统计校准）。

### 4.4 域内模型（CDnet，留出 4 类）
```bash
python3 train_real.py --holdout badWeather,nightVideos,turbulence,thermal ... --out checkpoints/gnet_real
```

---

## 5. 评测

### 5.0 单元测试（协议/GT 映射/因果背景）
```bash
PYTHONPATH=. python3 -m pytest -q tests   # 9 tests
```
覆盖：shadow/ignore 的 GT 映射、F1/PWC、类别平均、背景来源守卫与协议记录、
running/median 背景的因果性（前缀不变）与可靠性取值范围。

### 5.1 单次评测
```bash
# 冻结协议，因果背景
python3 eval.py --weights checkpoints/gnet_main_seed0/last.pt \
  --root datasets/cdnet/CDnet14 --bg-source causal --frame-stride 1 \
  --out logs/eval_x.csv
```

### 5.2 公平基线/消融（编排）
```bash
bash scripts/fair_protocol.sh     # SuBSENSE 地板 + 仅当前帧 + 主模型 3 seeds
bash scripts/fair_ablations.sh    # 7 个消融变体，stride1 causal
```

### 5.3 泄漏对照（背景来源切换）
```bash
bash scripts/leak_ablation.sh 0   # reconstructed / causal / firstframe / provided
```
输出 `logs/leak_*.csv`，用于 oracle gap。

### 5.3a RQ1 stride=1 复核（4 留出类）
```bash
bash scripts/leak_ablation_stride1.sh 1   # reconstructed/firstframe/provided，GPU1
```
输出 `logs/leak1_*.csv`；`causal` 列可直接由全量 causal 评测的 4 类汇总得到。
`scripts/leak_ablation.sh` 的 stride=3 版本仅用于旧表，不作为最终数字。

### 5.3b 免标注 teacher 基线（与 T4 同测试集）
```bash
python3 scripts/eval_teacher_holdout.py --stride 5 --workers 8   # CPU，约 10–15 分钟
```
输出 `logs/teacher_holdout_stride1.csv`：K=30、thr=35、translation GMC 的
因果中值 teacher 在 4 留出类上的 F1（默认 stride=5 以控制 CPU 成本；`--stride 1` 为完整协议，约 1–2 小时）。

### 5.4 传统基线（非学习残差阈值）
```bash
python3 diag_oracle_bg.py --stride 1 --out logs/diag_oracle_bg_stride1.json
python3 diag_oracle_bg.py --root datasets/sbi_root --stride 1 --out logs/diag_sbi.json
```

### 5.5 跨数据集（SBI 2015）
```bash
bash scripts/eval_sbi.sh 1
```

### 5.6 背景误差鲁棒性（E8）
```bash
bash scripts/e8_robust.sh          # stride=5，完整曲线（ghost/illum/noise）
bash scripts/e8_robust_stride1.sh  # stride=1，关键点（baseline/ghost.30/illum.20/noise.10）
```
输出 `logs/e8_*.csv`、`logs/e8s1_*.csv`。

### 5.7 效率/复杂度
```bash
python3 bench.py --weights checkpoints/gnet_main_seed0/last.pt --fusion cat --tsam-depth 12 --batch 1
```

---

## 6. 结果汇总

```bash
python3 scripts/report.py            # E1 / oracle gap / E3 消融 / E8 鲁棒性
python3 scripts/agg_results.py --gaps
```

关键结果文件（均为逐视频 CSV + 协议 sidecar JSON）：

| 论文位置 | 文件 |
|---|---|
| 主表（53 视频，stride 1） | `logs/fair_subsense.csv`, `logs/fair_cfo.csv`, `logs/fair_seed{0,1,2}_causal.csv` |
| 机制变体 | `logs/fair_m1_causal.csv`, `logs/fair_m3_causal.csv`, `logs/fair_full_causal.csv` |
| RQ1 泄漏表（stride 3 旧表） | `logs/leak_*.csv` |
| RQ1 stride=1 复核 | `logs/leak1_*.csv` |
| T4 学生 / teacher | `logs/fair_motion_holdout.csv`, `logs/teacher_holdout_stride1.csv` |
| 跨数据集 SBI | `logs/sbi_*.csv` |
| E8 鲁棒性 | `logs/e8_*.csv`, `logs/e8s1_*.csv` |
| 旧消融 stride=1 | `logs/fair_gnet_{notsam,d8,d16,nodda,nopmb,sub}_causal.csv` |

快速生成论文表：`python3 scripts/paper_tables.py`（RQ1 stride=1 行 + E8 关键点行，
缺失结果会标 `---`）。汇总数字同步到 `../E1_主对比表.md`。当前论文源码在 `paper/`，编译：
```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

> 注：`scripts/eval_causal_med.sh`（K=10 因果中值背景）**已弃用**：当前实现
> 按帧缓存 bg(1.2MB)+rel(1.7MB)，全量 118k 帧约需 340GB，会 OOM。L2a 结论
> 仅引用 `scripts/tune_bg.py` 的小样本结果（中值 vs EMA 残差 F1 0.619 vs 0.538）。

---

## 7. 目录结构

```
gnet_seg/
  train.py train_real.py eval.py eval_subsense.py bench.py diag_oracle_bg.py
  models/      gnet.py, tsam.py          # GNet + (M1 rel_gate) + (M3 boundary)
  data/        cdnet.py, synth.py, background.py, real.py, motion_labels.py
  utils/       metrics.py, protocol.py, boundary.py, losses.py
  scripts/     *.sh, report.py, agg_results.py
  checkpoints/ logs/ datasets/ third_party/
```

---

## 8. 需作者补充（复现缺口，见 `../实验可复现性核查.md`）

原稿未提供：数据/划分、训练超参、网络细节、SuBSENSE 参数、评测脚本、逐视频结果与权重。
本仓库的划分/超参为**复现自拟**，故结果与原稿数值不可直接对齐。

---

## 9. 计算量与许可

- 训练：8 epochs × 600 steps，batch 8，513²，单卡约 80 分钟。
- 全量评测：118,226 帧 @ ~27 样本/秒，单卡约 74 分钟。
- 数据许可：CDnet 2014（changedetection.net）、SBI（SBMI2015）、Cityscapes、COCO 2017 各自遵循其原始许可；本仓库不分发原始数据。
