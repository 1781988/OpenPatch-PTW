# OpenPatch-PTW

OpenPatch-PTW 是一个面向 **ICASSP 投稿实验** 的潜空间主动取证代码框架。项目在 GenPTW（AAAI 2026）的公开代码与 checkpoint 基础上，保留其 Watermark Encoder、CAF、原始 Spatial Fusion 主分支、消息解码器和篡改定位骨干，仅增加三部分：

1. **位置绑定空间代码（Position-Bound Spatial Code）**：由 64-bit 水印消息与二维坐标共同生成局部代码场；
2. **局部代码一致性图（Local Code Consistency Map）**：比较预测局部代码与理论代码，为小区域篡改定位提供额外线索；
3. **开放集状态判别（Open-Set Status Head）**：区分 `Unwatermarked / Valid / Forged`。

项目重点不是在原论文已接近饱和的标准 SD-Inpaint 指标上追求很小增益，而是系统评估：

- 1%–15% 小区域篡改；
- 无水印图像拒绝；
- 水印残差移植；
- 跨图局部水印移植；
- 未参与训练的同图 copy-move；
- 标准水印恢复与视觉质量是否保持。

> 当前仓库提供完整的环境、数据、训练、评估、基线、消融和结果打包流程。最终数值仍需在具备 COCO、官方模型权重和 GPU 的机器上实际运行后产生。

---

## 1. 原 GenPTW 是否开源实验数据

原 GenPTW 仓库没有直接发布一个包含全部论文实验样本的压缩数据包。其公开内容主要包括：

- COCO2017 train/val 的外部下载入口；
- Stable Diffusion 2 VAE；
- ConvNeXt-Tiny 权重；
- SD2 Inpainting；
- LaMA `big-lama.pt`；
- 官方 GenPTW checkpoint。

原论文描述的 1,000 张 SD2 生成图、UltraEdit 完整测试组合和 VAR 测试图没有在 GitHub 中作为统一数据包直接提供。因此，本项目采用以下可复现策略：

- 基础图像统一使用公开 COCO2017；
- 训练掩码和廉价篡改在运行时生成；
- 测试掩码由 `COCO image_id + 固定随机种子` 决定，重复运行一致；
- LaMA 和 SD-Inpaint 作为最终可选真实编辑测试，不作为快速训练的主要负担；
- 数据划分写入 manifest，避免每次运行样本不同。

---

## 2. 方法与风险控制

### 2.1 保留原 GenPTW 基线输出

新的 Spatial Fusion 不是直接删除原 SF。代码中保留了与原 GenPTW 完全对应的：

```text
Global projection + Base spatial fuse
```

官方 checkpoint 的 `watermark_2.proj` 和 `watermark_2.fuse` 会分别映射到：

```text
global_proj + base_fuse
```

新增位置分支使用零初始化残差：

```text
z_openpatch = z_genptw + sigmoid(alpha) * position_delta
```

因此，在成功载入官方 checkpoint 且位置分支尚未训练时，模型从原 GenPTW SF 起步，而不是从一个随机替换模块起步。

### 2.2 不修改 CAF1/CAF2

CAF1、CAF2 和消息编码器使用官方结构与权重，并默认冻结，减少水印恢复率和视觉质量突然下降的风险。

### 2.3 不扩展原 ConvNeXt 输入卷积

原 GenPTW 定位器使用：

```text
3-channel DCT high-frequency + 1-channel watermark feature
```

OpenPatch-PTW 不直接把其 4 通道 stem 改成随机的 5 通道 stem，而是保留官方 4 通道路径，并增加一个单独、零初始化的 `consistency_stem`。初始状态下定位结果与原分支兼容，随后只学习一致性图带来的增量信息。

### 2.4 确定性位置代码

位置代码生成器没有可训练参数，避免嵌入器和提取器共同退化成常量代码。理论代码可以由解出的水印消息与坐标直接重建。

### 2.5 训练攻击与留出攻击分离

默认训练只使用：

- residual transfer；
- cross-image patch transfer。

`copy-move` 仅用于测试，用于判断模型是否真正学习了位置一致性，而不是记忆某一种训练攻击。

---

## 3. 仓库结构

```text
OpenPatch-PTW/
├── configs/
│   ├── openpatch_ptw.yaml
│   └── ablations/
│       ├── no_position.yaml
│       ├── no_consistency.yaml
│       ├── no_status.yaml
│       └── fixed_mask.yaml
├── openpatch_ptw/
│   ├── attacks.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── data.py
│   ├── degradations.py
│   ├── genptw_bridge.py
│   ├── heads.py
│   ├── localizer.py
│   ├── losses.py
│   ├── masks.py
│   ├── metrics.py
│   ├── models.py
│   ├── position_code.py
│   ├── quality.py
│   ├── results.py
│   ├── runtime.py
│   └── visualize.py
├── scripts/
│   ├── bootstrap_genptw.sh
│   ├── download_coco.sh
│   ├── setup_env.sh
│   └── run_icassp_pipeline.sh
├── train_openpatch.py
├── eval_openpatch.py
├── prepare_data.py
├── download_assets.py
├── doctor.py
├── run_all_experiments.py
├── package_results.py
├── environment.yml
├── requirements.txt
└── tests/
```

---

## 4. 硬件与系统建议

推荐：

- Ubuntu 22.04；
- Python 3.10；
- NVIDIA GPU，至少 24 GB 显存；
- A100 40/80 GB、RTX 4090、RTX 3090 均可；
- COCO2017 约 26 GB；
- SD2 Inpainting 及缓存需要额外磁盘空间。

默认分辨率为 512×512，batch size 2，梯度累积 8。显存不足时先使用：

```bash
--override train.batch_size=1
```

不要优先降低分辨率，因为原 GenPTW checkpoint 与 512 分辨率设置关联较强。

---

## 5. 从零配置环境

### 5.1 拉取项目和上游代码

```bash
git clone https://github.com/1781988/OpenPatch-PTW.git
cd OpenPatch-PTW
bash scripts/bootstrap_genptw.sh
```

上游 GenPTW 会固定到本项目审查使用的 commit，并放在：

```text
third_party/GenPTW
```

### 5.2 自动创建环境

默认安装 PyTorch 2.7.0、TorchVision 0.22.0 和 CUDA 12.8 wheel：

```bash
bash scripts/setup_env.sh openpatch
conda activate openpatch
```

其他 CUDA wheel 可通过环境变量修改，例如：

```bash
CUDA_TAG=cu126 TORCH_VERSION=2.7.0 TORCHVISION_VERSION=0.22.0 \
  bash scripts/setup_env.sh openpatch
```

脚本先安装与 CUDA 匹配的 PyTorch，再安装 `requirements.txt` 和本项目 package，避免上游 requirements 中固定 CUDA wheel 与本机冲突。

---

## 6. 下载模型资产

### 6.1 最小训练资产

```bash
python download_assets.py \
  --config configs/openpatch_ptw.yaml \
  --components vae convnext checkpoint
```

目标目录为：

```text
third_party/GenPTW/
├── vae/
├── pytorch_model.bin
└── Checkpoint/
    ├── msg_decoder.pth
    ├── localizer.pth
    └── diffusion_pytorch_model.safetensors
```

### 6.2 最终真实编辑实验资产

```bash
python download_assets.py \
  --config configs/openpatch_ptw.yaml \
  --components lama sd_inpaint
```

如果 Hugging Face 模型需要登录：

```bash
export HF_TOKEN=你的令牌
python download_assets.py --components vae sd_inpaint
```

如果 Google Drive 自动下载官方 checkpoint 失败，按照 `third_party/GenPTW/README.md` 手动下载三个文件，并按上述目录放置。`doctor.py` 会检查是否完整。

---

## 7. 数据集如何使用

### 7.1 下载 COCO2017

默认下载到仓库内 `data/COCO2017`：

```bash
bash scripts/download_coco.sh data/COCO2017
```

最终目录必须是：

```text
data/COCO2017/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 7.2 AutoDL 推荐路径

大数据建议放到数据盘：

```bash
bash scripts/download_coco.sh /root/autodl-tmp/datasets/COCO2017
```

然后修改 `configs/openpatch_ptw.yaml`：

```yaml
data:
  train_img_dir: /root/autodl-tmp/datasets/COCO2017/train2017
  train_ann_file: /root/autodl-tmp/datasets/COCO2017/annotations/instances_train2017.json
  test_img_dir: /root/autodl-tmp/datasets/COCO2017/val2017
  test_ann_file: /root/autodl-tmp/datasets/COCO2017/annotations/instances_val2017.json
```

也可以不改文件，在命令行覆盖：

```bash
python prepare_data.py \
  --override data.train_img_dir=/root/autodl-tmp/datasets/COCO2017/train2017 \
  --override data.train_ann_file=/root/autodl-tmp/datasets/COCO2017/annotations/instances_train2017.json \
  --override data.test_img_dir=/root/autodl-tmp/datasets/COCO2017/val2017 \
  --override data.test_ann_file=/root/autodl-tmp/datasets/COCO2017/annotations/instances_val2017.json
```

### 7.3 创建固定划分

```bash
python prepare_data.py --config configs/openpatch_ptw.yaml
```

默认生成：

```text
data/manifests/
├── train_ids.txt   # 从 COCO train2017 中选择 50,000 张
├── dev_ids.txt     # 从 train2017 独立留出 1,000 张
├── test_ids.txt    # COCO val2017 的 5,000 张
└── dataset_report.json
```

训练集和 dev 不重叠。manifest 保存的是 COCO image ID，不是临时数组下标。

修改样本量：

```bash
python prepare_data.py \
  --override data.train_max_images=20000 \
  --override data.dev_max_images=500 \
  --override data.test_max_images=1000 \
  --force
```

### 7.4 是否需要提前生成篡改数据

核心实验不需要提前生成。训练时会在线生成：

- 多尺度掩码；
- plain-region replacement；
- residual transfer；
- cross-image patch transfer；
- 常见退化。

测试时根据固定 image ID 生成确定性掩码，因此结果可重复。LaMA 和 SD-Inpaint 在 `real_edits` suite 中按最终 checkpoint 在线生成，避免模型更新后缓存失效。

---

## 8. 运行前检查

只检查环境、路径、数据和权重：

```bash
python doctor.py --config configs/openpatch_ptw.yaml --strict
```

再执行一次完整前向验证：

```bash
python doctor.py --config configs/openpatch_ptw.yaml --strict --forward
```

输出：

```text
outputs/system/doctor_report.json
```

`--forward` 会实际运行 VAE 编码、plain/watermarked 解码、消息提取、局部代码、一致性图、状态头和定位器。如果这一步失败，不要开始长时间训练。

---

## 9. 三阶段训练

### 9.1 Smoke test

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --stage smoke \
  --max-steps 200
```

主要检查：

- 官方 checkpoint 是否真正加载；
- loss 是否有限；
- Bit ACC 是否没有直接退化到随机水平；
- `alpha`、mask F1、status accuracy 是否能更新；
- 可视化中水印图是否没有明显色偏。

### 9.2 新模块预热

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --stage warmup \
  --init-checkpoint outputs/icassp_main/smoke/best.pth
```

默认 5,000 个 optimizer steps。此阶段主要训练：

- position gate/residual；
- Local Code Head；
- Status Head；
- consistency stem；
- Mask Decoder。

### 9.3 联合微调

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --stage finetune \
  --init-checkpoint outputs/icassp_main/warmup/best.pth
```

默认 30,000 个 optimizer steps，并额外微调 ConvNeXt 后两级。原 VAE、CAF1/CAF2 和消息解码器默认保持冻结。

### 9.4 中断恢复

同一阶段中断后：

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --stage finetune \
  --resume outputs/icassp_main/finetune/step_12000.pth
```

`--resume` 恢复模型、优化器、scheduler、scaler 和 step；跨阶段初始化使用 `--init-checkpoint`，不要混用。

---

## 10. 完整评估

最终 checkpoint：

```text
outputs/icassp_main/finetune/best.pth
```

### 10.1 标准能力保持

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model openpatch \
  --suite standard \
  --checkpoint outputs/icassp_main/finetune/best.pth
```

包含：clean、noise、blur、brightness、contrast、resize、JPEG 90/70/50，以及 local splice。输出 Bit ACC、PSNR、SSIM、LPIPS、误报面积和局部定位指标。

### 10.2 小区域定位

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model openpatch \
  --suite small_tamper \
  --checkpoint outputs/icassp_main/finetune/best.pth
```

面积区间：

- 1%–5%；
- 5%–15%；
- 15%–30%；
- 30%–50%。

输出 F1、IoU、AUC、Boundary F1 和逐样本记录。

### 10.3 开放集来源验证

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model openpatch \
  --suite open_set \
  --checkpoint outputs/icassp_main/finetune/best.pth
```

三类：

- Unwatermarked；
- Valid；
- Forged。

输出 Macro-F1、Balanced Accuracy、Valid AUROC、EER、FAR@95%TPR 和混淆矩阵。

### 10.4 水印伪造与移植

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model openpatch \
  --suite forgery \
  --checkpoint outputs/icassp_main/finetune/best.pth
```

包含：

- residual transfer，`beta={0.5,0.75,1.0,1.25,1.5}`；
- cross-image patch transfer；
- held-out copy-move。

主要关注：

- Forgery Acceptance Rate；
- Attribution Success；
- 移植区域 F1/IoU；
- 不同 beta 下的安全性变化。

### 10.5 真实编辑

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model openpatch \
  --suite real_edits \
  --checkpoint outputs/icassp_main/finetune/best.pth
```

默认测试 LaMA 与 SD2 Inpainting，最多 200 张。该部分耗时显著高于其他 suite。

### 10.6 一次运行所有 suite

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model openpatch \
  --suite all \
  --checkpoint outputs/icassp_main/finetune/best.pth
```

---

## 11. 官方 GenPTW 同协议基线

同一 test manifest、掩码和攻击协议下运行：

```bash
python eval_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --run-name icassp_main \
  --model genptw \
  --suite standard

python eval_openpatch.py --config configs/openpatch_ptw.yaml --run-name icassp_main --model genptw --suite small_tamper
python eval_openpatch.py --config configs/openpatch_ptw.yaml --run-name icassp_main --model genptw --suite open_set
python eval_openpatch.py --config configs/openpatch_ptw.yaml --run-name icassp_main --model genptw --suite forgery
```

原 GenPTW 没有开放集状态头，因此其开放集基线使用平均 bit confidence 作为 valid score，并在合法水印样本上校准 95% TPR 阈值。这一点会在结果 JSON 中明确标注。

---

## 12. 消融实验

现有 variant：

| Variant | 作用 |
|---|---|
| `no_position.yaml` | 移除位置增量分支及一致性输入 |
| `no_consistency.yaml` | 保留位置嵌入，但不给定位器一致性图 |
| `no_status.yaml` | 移除开放集分类监督 |
| `fixed_mask.yaml` | 训练掩码固定回 15%–25% |

示例：

```bash
python train_openpatch.py \
  --config configs/openpatch_ptw.yaml \
  --variant configs/ablations/no_consistency.yaml \
  --run-name icassp_main_abl_no_consistency \
  --stage smoke \
  --max-steps 200
```

后续 warmup/finetune 与主模型相同，只需保持同一 `--variant`。

---

## 13. 一键实验流水线

### 13.1 快速链路验证

```bash
bash scripts/run_icassp_pipeline.sh quick icassp_quick
```

该模式使用很少 step 和样本，只验证代码链路，不用于论文结论。

### 13.2 核心 ICASSP 实验

```bash
bash scripts/run_icassp_pipeline.sh core icassp_main
```

包含：

- 数据 manifest；
- doctor；
- smoke、warmup、finetune；
- OpenPatch 四套核心评估；
- 官方 GenPTW 同协议基线；
- 自动结果打包。

### 13.3 论文完整模式

```bash
bash scripts/run_icassp_pipeline.sh paper icassp_main
```

在 core 基础上增加：

- LaMA / SD-Inpaint；
- 四组消融；
- 每组训练和评估；
- 统一结果包。

该模式计算量较大，建议先完成 core 并分析结果，再决定是否运行全部消融。

---

## 14. 输出目录

```text
outputs/
├── icassp_main/
│   ├── resolved_config.yaml
│   ├── environment.json
│   ├── upstream_load_report.json
│   ├── parameter_report.json
│   ├── smoke/
│   ├── warmup/
│   ├── finetune/
│   ├── eval/
│   │   ├── openpatch/
│   │   │   ├── standard/
│   │   │   ├── small_tamper/
│   │   │   ├── open_set/
│   │   │   ├── forgery/
│   │   │   └── real_edits/
│   │   └── genptw/
│   └── pipeline_logs/
└── bundles/
```

每个评估 suite 均输出：

- 聚合 JSON；
- 逐样本 CSV；
- 逐样本 JSONL；
- 有限数量可视化；
- 按攻击或面积区间统计的均值、标准差、中位数、最小值、最大值。

---

## 15. 打包实验数据并上传分析

手动打包：

```bash
python package_results.py \
  --config configs/openpatch_ptw.yaml \
  --run-prefix icassp_main
```

输出类似：

```text
outputs/bundles/icassp_main_20260817TxxxxxxZ.zip
```

默认包含：

- `paper_summary.csv`；
- 全部评估 JSON/CSV/JSONL；
- 训练与验证日志；
- 配置和环境报告；
- checkpoint 加载报告；
- 代表性可视化；
- `ANALYSIS_GUIDE.md`。

默认不包含：

- COCO 图像；
- SD/LaMA 模型；
- 大 checkpoint。

需要把 checkpoint 也打包时：

```bash
python package_results.py \
  --config configs/openpatch_ptw.yaml \
  --run-prefix icassp_main \
  --include-checkpoints
```

通常上传分析时只需要默认 zip。逐样本 CSV 能够进一步分析：

- 哪个面积区间失败最多；
- 哪种 mask shape 最难；
- copy-move 与 cross-image 的差异；
- 视觉质量和局部定位之间是否存在冲突；
- 状态头是否把合法局部编辑误判为 Forged。

---

## 16. 建议的论文验收标准

这些是决策阈值，不是预先宣称的实验结果：

- 相比官方 GenPTW，PSNR 下降不超过约 0.8 dB；
- 标准 Bit ACC 下降不超过约 1 个百分点；
- 1%–5% 和 5%–15% 定位 F1/Boundary F1 有稳定提升；
- Valid vs non-valid AUROC 达到可用水平；
- Forgery Acceptance Rate 显著低于 bit-confidence 基线；
- held-out copy-move 仍有明显识别或定位收益。

如果开放集结果不稳定、但小区域定位稳定提升，可将论文收缩为位置绑定的细粒度篡改定位；如果定位提升有限、但伪造拒绝明显，则转为开放集来源验证主线。

---

## 17. 常见问题

### 17.1 官方 SF 权重没有加载

`doctor --forward` 或训练会直接报错。检查：

```text
third_party/GenPTW/Checkpoint/diffusion_pytorch_model.safetensors
```

并查看：

```text
outputs/<run>/upstream_load_report.json
```

`base_sf_loaded` 和 `caf_loaded` 不应为 0。

### 17.2 OOM

先使用：

```bash
--override train.batch_size=1 \
--override train.gradient_accumulation_steps=16
```

然后可暂时关闭 LPIPS 做链路检查：

```bash
--override quality.use_lpips=false
```

最终正式训练应恢复 LPIPS。

### 17.3 Google Drive checkpoint 下载失败

手动按照上游 README 下载三个 checkpoint 文件。不要把三个文件改名。

### 17.4 real_edits 很慢

先限制：

```bash
--max-samples 20
```

确认流程正确后，再逐步扩展到 200。

### 17.5 训练日志中的 mask F1 在 clean batch 为 0

clean valid 样本的 GT mask 全零，单独的前景 F1 不适合作为该 batch 的主要指标。最终定位结论以 `small_tamper`、`local_splice` 和 `real_edits` 聚合结果为准。

---

## 18. 许可证与引用

GenPTW 上游使用 Non-Commercial Research License。本仓库采用相同的非商业研究限制，详细信息见：

- `LICENSE.txt`；
- `THIRD_PARTY_NOTICE.md`。

发表论文时必须引用原 GenPTW 工作，并清楚说明本方法基于其公开实现和 checkpoint 开展增量研究。
