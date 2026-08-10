# OpenPatch-PTW

> 面向开放集来源验证与小区域篡改定位的位置绑定潜空间水印框架。基于 **GenPTW (AAAI 2026)** 的公开代码与公开 checkpoint 进行增量改造，目标是以较低工程成本验证两项改进：**位置绑定式空间水印（Position-Bound Spatial Code）** 与 **开放集真实性判别（Unwatermarked / Valid / Forged）**。

## 1. 项目定位

GenPTW 已实现“来源追踪 + 篡改定位”的统一潜空间水印。OpenPatch-PTW 不重写 CAF、DCT、ConvNeXt 等成熟模块，而是尽量保留原始框架，只针对其更适合快速发表/落地的两个缺口进行扩展：

1. 原 Spatial Fusion 将同一个全局水印向量广播到所有空间位置，缺少显式的位置身份；本项目使用 **水印消息 + 二维 Fourier 坐标** 生成位置相关空间代码场。
2. 原水印解码器默认输入图像已含目标水印；本项目增加 **三分类状态头**，区分无水印、合法水印、伪造/移植水印。
3. 原训练掩码偏向中等面积目标；本项目增加 1%–50% 多尺度掩码、copy-move、跨图块移植、残差移植，重点评估小区域定位与伪造拒绝。

本仓库采用“**上游 GenPTW + 本仓库增量模块**”的方式组织，避免复制大量上游代码，也便于直接复用官方 checkpoint。

---

## 2. 原论文数据是否开源？

**原 GenPTW 仓库没有把完整实验数据直接存入 GitHub。** 上游 README 提供的是外部公开数据/模型的下载入口：

- 训练图像与分割标注：**MS COCO 2017 train**；
- 基础测试集：**MS COCO 2017 val**；
- Stable Diffusion 2 VAE；
- ConvNeXt-Tiny 预训练权重；
- Stable Diffusion 2 Inpainting；
- LaMA `big-lama.pt`；
- GenPTW 官方 checkpoint（Google Drive）。

论文实验还描述了 5,000 张 COCO 验证图、由 Stable Diffusion v2 根据 COCO captions 生成的 1,000 张 AI 图像、UltraEdit 编辑协议以及 VAR 的 1,000 张图像，但这些“论文中生成/整理后的完整测试样本集合”并未在上游 GitHub 仓库中作为数据包直接发布。因此，本项目默认使用可公开下载的 COCO2017，并在本地按固定随机种子生成本项目所需的篡改/伪造样本。

---

## 3. 目录结构

```text
OpenPatch-PTW/
├── configs/
│   └── openpatch_ptw.yaml          # 默认实验配置
├── openpatch_ptw/
│   ├── __init__.py
│   ├── position_code.py            # Fourier 坐标编码、位置代码场、位置绑定 SF
│   ├── heads.py                    # 局部代码解码头、三分类状态头
│   ├── localizer.py                # 5 通道定位器：高频图 + WM 特征 + 一致性图
│   ├── attacks.py                  # 残差移植、跨图块移植、同图 copy-move
│   ├── masks.py                    # 1%–50% 多尺度/多形状掩码
│   ├── losses.py                   # code/status/Dice 等新增损失
│   ├── metrics.py                  # F1/IoU/AUROC/FAR 等
│   └── genptw_bridge.py            # 将增量模块注入上游 GenPTW VAE
├── scripts/
│   ├── bootstrap_genptw.sh         # 拉取上游 GenPTW
│   └── download_coco.sh            # 下载 COCO2017 train/val/annotations
├── train_openpatch.py              # 分阶段微调入口
├── eval_openpatch.py               # 标准/开放集/伪造评估入口
├── requirements.txt
├── THIRD_PARTY_NOTICE.md
└── LICENSE.txt
```

---

## 4. 环境配置

### 4.1 推荐环境

- Linux / Ubuntu 22.04 或 AutoDL 同类环境
- Python 3.10
- NVIDIA GPU，建议 >= 24 GB 显存；A100/4090/3090 均可
- CUDA 版本以 PyTorch 安装包为准

### 4.2 获取代码

```bash
git clone https://github.com/1781988/OpenPatch-PTW.git
cd OpenPatch-PTW
bash scripts/bootstrap_genptw.sh
```

脚本会把上游仓库放到：

```text
third_party/GenPTW
```

### 4.3 安装依赖

优先保持与上游代码一致：

```bash
conda create -n openpatch python=3.10 -y
conda activate openpatch

pip install -r third_party/GenPTW/requirements.txt
pip install -r requirements.txt
```

如果上游 `torch==...+cu...` 与本机 CUDA 不匹配，请先根据本机 CUDA 安装对应 PyTorch，再安装其余依赖：

```bash
# 示例：先安装与你机器匹配的 torch/torchvision
# 然后跳过上游 requirements 中 torch/torchvision 两行，再安装其余包。
```

---

## 5. 数据配置

### 5.1 COCO2017

自动下载：

```bash
bash scripts/download_coco.sh /data/COCO2017
```

得到：

```text
/data/COCO2017/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

也可以手动从 COCO 官方地址下载。原 GenPTW 训练同样使用 COCO2017 图像和实例分割标注。

### 5.2 上游模型/checkpoint

按照 `third_party/GenPTW/README.md` 下载并组织：

```text
third_party/GenPTW/
├── vae/                         # Stable Diffusion 2 Base VAE
├── big-lama.pt                  # LaMA
├── pytorch_model.bin            # ConvNeXt-Tiny
└── Checkpoint/
    ├── msg_decoder.pth
    ├── localizer.pth
    └── diffusion_pytorch_model.safetensors
```

本仓库不重新分发上述大文件。

### 5.3 修改配置

编辑 `configs/openpatch_ptw.yaml`：

```yaml
data:
  train_img_dir: /data/COCO2017/train2017
  train_ann_file: /data/COCO2017/annotations/instances_train2017.json
  val_img_dir: /data/COCO2017/val2017
  val_ann_file: /data/COCO2017/annotations/instances_val2017.json

upstream:
  root: third_party/GenPTW
  vae: third_party/GenPTW/vae
  checkpoint_dir: third_party/GenPTW/Checkpoint
```

---

## 6. 方法实现

### 6.1 Position-Bound Spatial Code

原始 SF 的空间先验在所有位置相同。OpenPatch-PTW 使用：

```text
64-bit watermark -> message embedding ─┐
                                      ├─> PositionCodeField -> C(u,v)
2D Fourier coordinate encoding -------┘
```

代码场 `C(u,v)` 与水印消息和坐标同时相关，同一水印在不同位置具有不同局部表示。随后使用小尺度门控残差注入：

```text
z_wm = z + sigmoid(alpha) * gate(z, C) * residual(z, C)
```

`alpha` 小值初始化，降低新模块破坏图像质量的风险。

### 6.2 Local Code Consistency

提取端从 GenPTW 的共享水印特征预测局部代码 `C_hat`；再根据解出的水印比特重新生成期望代码 `C_exp`：

```text
R = mean(abs(C_hat - C_exp), channel)
```

`R` 为局部一致性残差图。定位器输入由原来的：

```text
DCT high-frequency (3ch) + WM feature (1ch)
```

扩展为：

```text
DCT high-frequency (3ch) + WM feature (1ch) + consistency map (1ch)
```

### 6.3 Open-Set Status Head

状态头输出三类：

- `0 = Unwatermarked`：无目标水印；
- `1 = Valid`：合法水印，包括正常图和来源合法但被局部编辑的图；
- `2 = Forged`：跨图移植、残差移植、异位置复制等伪造样本。

最终业务输出可组合为：无水印 / 合法完整 / 合法但被篡改 / 水印伪造。

---

## 7. 风险规避设计

本实现有意避免几类高风险方案：

1. **不做密码学签名/HMAC**：避免密钥管理与安全证明拖慢实验。
2. **不做语义内容哈希**：避免“内容绑定”在不同编辑下难以稳定的问题。
3. **不修改 CAF1/CAF2**：最大限度复用 GenPTW 已验证模块和 checkpoint。
4. **小幅注入 + 分阶段训练**：先只训练新 SF / code head / status head，再联合微调 decoder/localizer 后半部分。
5. **伪造训练与测试攻击分离**：例如训练 residual-transfer + cross-image paste，测试时额外保留 copy-move，减少分类头仅记忆训练攻击的风险。
6. **标准任务必须保持**：新方法首先保证 PSNR、Bit ACC、标准 SD-Inpaint/LaMA 定位不明显退化，再强调开放集和小篡改提升。

---

## 8. 训练设置

默认配置：

```yaml
model:
  bit_dim: 64
  code_dim: 8
  fourier_bands: 4
  resolution: 512

train:
  seed: 2026
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  warmup_new_modules_steps: 5000
  finetune_steps: 30000
  mixed_precision: bf16

sample_mix:
  valid: 0.50
  unwatermarked: 0.25
  forged: 0.25
```

### 8.1 快速 smoke test

先在少量数据上确认框架可运行：

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --stage smoke \
  --max_steps 200
```

### 8.2 阶段一：只训练新增模块

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --stage warmup \
  --max_steps 5000
```

建议冻结：

- 原始 VAE；
- CAF1/CAF2；
- GenPTW message decoder 主干；
- ConvNeXt 大部分层。

训练：

- Position-Bound SF；
- Local Code Head；
- Status Head；
- Localizer 新增第 5 输入通道及末端解码层。

### 8.3 阶段二：联合微调

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --stage finetune \
  --max_steps 30000
```

如果显存/时间有限，可把真实 LaMA/SD-Inpaint 在线攻击比例控制在 20%，其余使用 residual-transfer、paste、copy-move 等低成本合成攻击。

---

## 9. 评估设置

### 9.1 标准能力保持

```bash
python eval_openpatch.py --config configs/openpatch_ptw.yaml --suite standard
```

建议报告：

- PSNR / SSIM / LPIPS；
- Bit ACC；
- SD Inpaint / LaMA / Splicing 的 F1、IoU、AUC。

### 9.2 小区域篡改

```bash
python eval_openpatch.py --config configs/openpatch_ptw.yaml --suite small_tamper
```

固定区间：

- 1%–5%；
- 5%–15%；
- 15%–30%；
- 30%–50%。

报告 F1 / IoU / AUC，并至少给出 1%–5% 的可视化案例。

### 9.3 开放集来源验证

```bash
python eval_openpatch.py --config configs/openpatch_ptw.yaml --suite open_set
```

测试组成建议：

- 1,000 合法水印图；
- 1,000 无水印自然图；
- 1,000 无水印 AI 图；
- 1,000 伪造/移植图。

报告：Macro-F1、AUROC、FAR@95%TPR、混淆矩阵。

### 9.4 伪造/移植攻击

```bash
python eval_openpatch.py --config configs/openpatch_ptw.yaml --suite forgery
```

测试：

- residual transfer；
- cross-image patch transfer；
- same-image copy-move；
- 不同篡改面积；
- residual scaling `beta in {0.5, 0.75, 1.0, 1.25, 1.5}`。

主要指标：

- Forgery Acceptance Rate；
- Attribution Attack Success Rate；
- 移植区域 F1/IoU。

---

## 10. 推荐论文消融

| 版本 | 多尺度掩码 | 位置代码 | 一致性图 | 三分类头 | 伪造训练 |
|---|---:|---:|---:|---:|---:|
| GenPTW | × | × | × | × | × |
| + Multi-scale | ✓ | × | × | × | × |
| + Position Code | ✓ | ✓ | × | × | × |
| + Consistency | ✓ | ✓ | ✓ | × | × |
| OpenPatch-PTW | ✓ | ✓ | ✓ | ✓ | ✓ |

另外测试 `code_dim in {4, 8, 16}`，默认 8。

---

## 11. 建议的论文成功判据

以下是项目验收目标，不是预先宣称的结果：

- 相比 GenPTW，PSNR 下降不超过约 0.8 dB；
- Bit ACC 下降不超过约 1 个百分点；
- 1%–5% 小区域定位 F1 有明确提升；
- Valid vs. Non-valid AUROC 达到可用水平；
- 伪造接受率显著低于“只根据 Bit ACC 判断是否合法”的基线。

若开放集分类表现一般但小区域定位提升稳定，可退化为“Position-Bound GenPTW + Multi-scale Tamper Localization”版本投稿；若定位提升不明显但伪造拒绝很强，则转为“Open-Set Provenance Verification”主线。

---

## 12. 许可证与引用

GenPTW 上游代码采用 **Non-Commercial Research License**，其衍生工作也要求保持非商业研究限制。本仓库因此采用相同性质的非商业研究许可，并在 `THIRD_PARTY_NOTICE.md` 中保留上游来源与引用。

如果本项目用于论文，请同时引用 GenPTW 原论文。
