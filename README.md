# Oxford-IIIT Pet 分类与前景分割实验

## 项目目录结构

```text
data/
src/
  config.py
  data_transforms.py
  data_loaders.py
  inspect_data.py
  inspect_preprocessing.py
  check_dataloaders.py
outputs/
  checkpoints/
  logs/
  figures/
  results/
main.tex
执行步骤.md
方案.md
```

## 关键默认配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 随机种子 | `42` | 用于数据划分和训练初始化 |
| 数据目录 | `data/oxford-iiit-pet` | Oxford-IIIT Pet 数据缓存目录 |
| 分类输入尺寸 | `224` | ResNet18 与 EfficientNet-B0 输入尺寸 |
| 分割输入尺寸 | `320` | DeepLabV3-MobileNetV3 输入尺寸 |
| 分类 batch size | `32` | 显存不足时可降为 `16` |
| 分割 batch size | `4` | 显存不足时可降为 `2` |
| 输出目录 | `outputs/` | 统一保存权重、日志、图片和结果 |

这些默认值集中定义在 `src/config.py` 中，后续训练、评估和可视化脚本应通过命令行参数覆盖默认配置。

## 当前进度记录

### Step 3.3 预处理结果抽查

已使用 `src/inspect_preprocessing.py` 对分类和分割预处理结果进行抽查。

产物：

- `outputs/figures/fig_preprocessing_check.png`：原图、分类 train / val 预处理图、原始 trimap、分割 train 图像与 mask、分割 val mask 对比。
- `outputs/results/preprocessing_check.json`：样本索引、张量形状、原始与处理后 mask 像素值记录。

结论：分类输出尺寸为 `[3, 224, 224]`，分割图像输出尺寸为 `[3, 320, 320]`，分割 mask 输出尺寸为 `[320, 320]`。当前阶段 mask 保持原始 trimap 整数值 `[1, 2, 3]`，未出现小数或异常类别值；二值前景 mask 转换将在 Step 4 执行。

### Step 3.4 DataLoader 批量输出检查

已补充 `src/data_loaders.py` 和 `src/check_dataloaders.py`，将 Step 3 的 transform 固化为后续训练可复用的 batch 接口。

产物：

- `src/data_loaders.py`：封装分类与分割 Dataset/DataLoader，使用固定随机种子把官方 `trainval` 划分为 train / val，官方 `test` 保持为最终测试集。
- `src/check_dataloaders.py`：检查分类与分割 dataloader 的 batch 形状、dtype、标签范围和 mask 像素值。
- `outputs/results/dataloader_check.json`：batch 级检查记录。

检查命令：

```powershell
D:\Anaconda3\envs\unet\python.exe src\check_dataloaders.py --cls-batch-size 4 --seg-batch-size 2 --num-workers 0
```

当前检查结果：

| 任务 | split | 数据量 | batch 输出 |
| --- | --- | ---: | --- |
| 分类 | train | 2944 | images `[4, 3, 224, 224]`，labels `[4]` |
| 分类 | val | 736 | images `[4, 3, 224, 224]`，labels `[4]` |
| 分类 | test | 3669 | images `[4, 3, 224, 224]`，labels `[4]` |
| 分割 | train | 2944 | images `[2, 3, 320, 320]`，masks `[2, 320, 320]`，values `[1, 2, 3]` |
| 分割 | val | 736 | images `[2, 3, 320, 320]`，masks `[2, 320, 320]`，values `[1, 2, 3]` |
| 分割 | test | 3669 | images `[2, 3, 320, 320]`，masks `[2, 320, 320]`，values `[1, 2, 3]` |

结论：Step 3 的分类和分割预处理、同步空间变换、mask 最近邻插值、batch 输出接口均已通过检查。当前分割 mask 仍保留原始 trimap 值 `[1, 2, 3]`，符合“Step 4 再做二值 mask 转换”的流程。
