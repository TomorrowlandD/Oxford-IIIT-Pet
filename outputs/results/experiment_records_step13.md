# Step 13 实验过程原始记录

## 数据读取记录

- 数据集：Oxford-IIIT Pet。
- 类别数量：37 类宠物品种。
- 随机种子：42。
- 划分方式：官方 `trainval` 按 8:2 固定划分为训练集和验证集，官方 `test` 仅用于最终测试。
- 样本数量：官方 `trainval` 为 3680 张，训练集 2944 张，验证集 736 张，测试集 3669 张。
- 数据统计文件：`outputs/results/data_summary.json`。
- 划分索引文件：`outputs/results/split_indices_seed42.json`。

## Mask 转换记录

- 原始 trimap 像素值：`1`、`2`、`3`。
- 像素含义：`1` 为宠物前景，`2` 为背景，`3` 为边界。
- 采用规则：方案 B，将边界并入前景。
- 二值映射：原始 `1 -> 1`，`2 -> 0`，`3 -> 1`。
- 输出类别：背景 `0`，前景 `1`。
- 损失函数和指标口径：普通二分类 `CrossEntropyLoss`，指标按 `0/1` 标签计算 Pixel Accuracy、背景 IoU、前景 IoU、mIoU 和 Dice。
- 规则记录：`outputs/results/trimap_rule_step4.json`。
- 转换抽查：`outputs/results/mask_conversion_check.json`。
- 样例图：`outputs/figures/fig_mask_samples.png`。

## 实验编号映射

| 实验编号 | 任务 | 模型 | 输入尺寸 | batch size | 训练 epoch | 最佳 epoch | 学习率 | 训练日志 | 最佳权重 | 结果文件 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Exp-1 | 分类 | ResNet18 | 224 | 32 | 20 | 15 | 1e-4 | `outputs/logs/resnet18_cls_log.csv` | `outputs/checkpoints/best_cls_resnet18.pth` | `outputs/results/cls_metrics_resnet18.json` |
| Exp-2 | 分类 | EfficientNet-B0 | 224 | 32 | 25 | 13 | 1e-4 | `outputs/logs/efficientnet_b0_cls_log.csv` | `outputs/checkpoints/best_cls_efficientnet_b0.pth` | `outputs/results/cls_metrics_efficientnet_b0.json` |
| Exp-3 | 分割 | DeepLabV3-MobileNetV3 | 320 | 4 | 30 | 23 | 1e-4 | `outputs/logs/deeplabv3_mobilenet_seg_log.csv` | `outputs/checkpoints/best_seg_deeplabv3_mobilenet.pth` | `outputs/results/seg_metrics_deeplabv3_mobilenet.json` |

## 训练命令记录

Exp-1 ResNet18 分类 baseline：

```powershell
D:\Anaconda3\envs\unet\python.exe src\train_cls.py --model resnet18 --img-size 224 --batch-size 32 --epochs 20 --lr 1e-4 --pretrained
```

Exp-2 EfficientNet-B0 分类主模型：

```powershell
D:\Anaconda3\envs\unet\python.exe src\train_cls.py --model efficientnet_b0 --img-size 224 --batch-size 32 --epochs 25 --lr 1e-4 --pretrained --amp --num-workers 0
```

Exp-3 DeepLabV3-MobileNetV3 前景分割模型：

```powershell
D:\Anaconda3\envs\unet\python.exe src\train_seg.py --model deeplabv3_mobilenet --data-root data\step6_data --img-size 320 --batch-size 4 --epochs 30 --lr 1e-4 --pretrained --binary-mask
```

## 最终输出记录

| 实验编号 | 最终测试指标摘要 | 辅助输出 |
| --- | --- | --- |
| Exp-1 | Accuracy `0.883074`，Macro-F1 `0.881759`，Top-5 Accuracy `0.986645` | `outputs/results/cls_predictions_resnet18_test.csv`，`outputs/figures/fig_confusion_matrix_resnet18.png` |
| Exp-2 | Accuracy `0.899700`，Macro-F1 `0.898173`，Top-5 Accuracy `0.995639` | `outputs/results/cls_predictions_efficientnet_b0_test.csv`，`outputs/figures/fig_confusion_matrix_efficientnet_b0.png`，`outputs/figures/fig_cls_correct_examples.png`，`outputs/figures/fig_failure_cases.png` |
| Exp-3 | Pixel Accuracy `0.961981`，mIoU `0.924921`，Dice `0.954638` | `outputs/results/seg_predictions_deeplabv3_mobilenet_test.csv`，`outputs/figures/fig_seg_predictions.png` |

## 可视化材料清单

- 数据样例：`outputs/figures/fig_dataset_samples.png`。
- 类别分布：`outputs/figures/fig_class_distribution.png`。
- mask 样例：`outputs/figures/fig_mask_samples.png`。
- 分类训练曲线：`outputs/figures/fig_cls_training_curve.png`。
- EfficientNet-B0 混淆矩阵：`outputs/figures/fig_confusion_matrix.png`。
- ResNet18 混淆矩阵：`outputs/figures/fig_confusion_matrix_resnet18.png`。
- 分割训练曲线：`outputs/figures/fig_seg_training_curve.png`。
- 分割预测对比：`outputs/figures/fig_seg_predictions.png`。
- 分类正确样例：`outputs/figures/fig_cls_correct_examples.png`。
- 失败案例：`outputs/figures/fig_failure_cases.png`。

## 记录结论

三个核心实验均可以由实验编号追溯到训练命令、CSV 日志、最佳权重和最终测试结果。分类任务中 EfficientNet-B0 在官方测试集上优于 ResNet18，可作为主模型；ResNet18 保留为 baseline。分割任务中 DeepLabV3-MobileNetV3 在官方测试集上取得 mIoU `0.924921` 和 Dice `0.954638`，并已生成成功与困难样例用于报告分析。
