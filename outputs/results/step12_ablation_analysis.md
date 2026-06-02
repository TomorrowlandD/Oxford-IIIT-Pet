# Step 12 轻量消融实验：ResNet18 vs EfficientNet-B0

## 实验设置

本消融实验复用分类任务的两个已训练模型，比较 ResNet18 baseline 与 EfficientNet-B0 主模型在 Oxford-IIIT Pet 官方测试集上的表现。两者使用相同的数据划分、相同测试集、相同输入尺寸 `224 x 224`，并均使用 ImageNet 预训练权重进行迁移学习微调。

训练集、验证集和测试集划分保持固定：官方 `trainval` 按随机种子 `42` 划分为训练集 `2944` 张和验证集 `736` 张，官方 `test` 集 `3669` 张仅用于最终评估。

## 测试集结果

| 模型 | 最佳 epoch | 验证集 Accuracy | 验证集 Macro-F1 | 测试集 Accuracy | 测试集 Macro-F1 | Top-5 Accuracy | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 15 | 0.930707 | 0.930059 | 0.883074 | 0.881759 | 0.986645 | 765.40 s |
| EfficientNet-B0 | 13 | 0.937500 | 0.935778 | 0.899700 | 0.898173 | 0.995639 | 1010.81 s |

与 ResNet18 相比，EfficientNet-B0 在测试集上 Accuracy 提升 `0.016626`，Macro-F1 提升 `0.016414`，Top-5 Accuracy 提升 `0.008994`，测试 loss 降低 `0.073923`。这说明 EfficientNet-B0 在整体分类性能和候选类别排序上都更稳健。

## 类别层面对比

EfficientNet-B0 对若干容易混淆的类别有明显改善。F1 提升较大的类别包括：

| 类别 | ResNet18 F1 | EfficientNet-B0 F1 | F1 提升 |
| --- | ---: | ---: | ---: |
| Ragdoll | 0.694836 | 0.765957 | +0.071122 |
| Birman | 0.758294 | 0.822430 | +0.064136 |
| Staffordshire Bull Terrier | 0.586207 | 0.648649 | +0.062442 |
| Abyssinian | 0.835979 | 0.897561 | +0.061582 |
| Pomeranian | 0.891089 | 0.934010 | +0.042921 |

不过 EfficientNet-B0 并非所有类别都优于 ResNet18。Maine Coon、Yorkshire Terrier、German Shorthaired、Egyptian Mau 等类别的 F1 略有下降，说明主模型虽然整体更强，但局部类别仍受样本姿态、毛色纹理和相似外观影响。

## 混淆情况

两个模型的主要错误都集中在外观相近的品种之间：

- American Pit Bull Terrier 被误判为 Staffordshire Bull Terrier：ResNet18 为 `20` 例，EfficientNet-B0 为 `24` 例。
- American Pit Bull Terrier 被误判为 American Bulldog：ResNet18 为 `16` 例，EfficientNet-B0 为 `13` 例。
- Ragdoll 与 Birman 存在双向混淆：ResNet18 中 Ragdoll -> Birman 为 `15` 例，Birman -> Ragdoll 为 `16` 例；EfficientNet-B0 中 Ragdoll -> Birman 为 `14` 例，Birman -> Ragdoll 为 `10` 例。
- Siamese 被误判为 Birman：ResNet18 为 `13` 例，EfficientNet-B0 为 `11` 例。

这些错误与宠物品种分类任务本身的细粒度性质一致。相近犬种之间头部轮廓、体型和毛色差异较小；相近猫种之间毛色、眼睛和脸部纹理容易受拍摄角度、遮挡和背景影响。

## 结论

EfficientNet-B0 在本实验中优于 ResNet18，测试集 Accuracy 从 `0.883074` 提升到 `0.899700`，Macro-F1 从 `0.881759` 提升到 `0.898173`。考虑到 EfficientNet-B0 的训练总耗时约为 `1010.81 s`，高于 ResNet18 的 `765.40 s`，该提升是以一定训练时间成本换来的。

综合结果看，EfficientNet-B0 更适合作为本项目分类任务主模型；ResNet18 结构简单、训练更快，适合作为 baseline。两者共同说明：在 Oxford-IIIT Pet 这类细粒度品种分类任务中，更强的特征提取结构能改善整体性能，但相似品种之间的混淆仍然是主要误差来源。

## 对应产物

- 测试集消融汇总 CSV：`outputs/results/cls_ablation_test_summary.csv`
- 测试集消融汇总 JSON：`outputs/results/cls_ablation_test_summary.json`
- 验证集训练汇总：`outputs/results/cls_ablation_val_summary.csv`
- 分类训练曲线：`outputs/figures/fig_cls_training_curve.png`
- 混淆矩阵：`outputs/figures/fig_confusion_matrix.png`
