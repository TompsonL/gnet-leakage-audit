# GNet 效率/复杂度基准

设备：NVIDIA L20 (46GB)，输入 513×513，PyTorch 2.13.0+cu130，FP32 前向。
注：测量时 GPU0 同时运行公平协议评测，吞吐为**并发下的下界**；空闲时数值会略高。

## 网络（双输入当前帧+背景）

| 配置 | 参数量(M) | GMACs | GFLOPs | 延迟(ms/张) | 吞吐(FPS) | 峰值显存(GB) |
|---|---|---|---|---|---|---|
| main d12 fusion=cat, bs1 | 60.58 | 431.7 | 863.5 | 62.0 | 16.1 | 0.78 |
| main d12 fusion=cat, bs8 | 60.58 | 431.7 | 863.5 | 75.4 | 13.3 | 4.42 |
| d12 fusion=sub, bs1 | 60.24 | 430.2 | 860.3 | 60.2 | 16.6 | 0.74 |
| TSAM 8, bs1 | 57.42 | 431.4 | 862.8 | 61.4 | 16.3 | 0.77 |
| TSAM 16, bs1 | 63.74 | 432.1 | 864.2 | 62.6 | 16.0 | 0.80 |
| 无 TSAM (depth1), bs1 | 51.89 | 430.8 | 861.5 | 59.6 | 16.8 | 0.75 |

结论：**FLOPs 基本由 ResNet101 双路主干决定（~430 GMACs），与 TSAM 深度几乎无关**；TSAM 只影响参数量（51.9→63.7M）与少量时延。这说明“用 12 层 Transformer 聚合”在复杂度上并非瓶颈，但也说明 TSAM 的边际收益有限。

## 背景估计（CPU，因果、无掩码）

在 nightVideos/bridgeEntry（2500 帧）上测：

| 方法 | 总耗时 | 每帧 |
|---|---|---|
| 因果 running 背景（M1 基础） | 4.33 s | 1.73 ms |
| 因果背景 + 可靠性图（M1 完整） | 10.61 s | 4.24 ms |

## 整流程（单流，因果）

网络 62 ms + 可靠性背景 4.2 ms（CPU 可与网络并行）≈ **网络主导，单流约 15 FPS**；
背景估计 CPU 占用低，可异步/多流并行。满足在线因果推理（仅用过去帧）。

## 复现命令

```
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
python3 bench.py --weights checkpoints/gnet_main_seed0/last.pt --fusion cat --tsam-depth 12 --batch 1 --iters 30 --label main
```
