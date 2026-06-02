# Step 11.3 结果分析类图像记录

## 已生成图像

- 分类混淆矩阵：`outputs/figures/fig_confusion_matrix.png`
- EfficientNet-B0 分类正确样例：`outputs/figures/fig_cls_correct_examples.png`
- EfficientNet-B0 分类失败案例：`outputs/figures/fig_failure_cases.png`
- DeepLabV3-MobileNetV3 分割预测对比：`outputs/figures/fig_seg_predictions.png`

## 分类混淆矩阵分析要点

分类混淆矩阵使用 EfficientNet-B0 在官方 test split 上的最终预测结果生成。测试集共有 3669 张图像，Accuracy 为 0.899700，Macro-F1 为 0.898173。

主要混淆对如下：

| 真实类别 | 预测类别 | 错误数量 |
| --- | --- | ---: |
| American Pit Bull Terrier | Staffordshire Bull Terrier | 24 |
| Ragdoll | Birman | 14 |
| American Pit Bull Terrier | American Bulldog | 13 |
| Egyptian Mau | Bengal | 12 |
| Siamese | Birman | 11 |
| Staffordshire Bull Terrier | American Bulldog | 10 |
| Birman | Ragdoll | 10 |
| Staffordshire Bull Terrier | American Pit Bull Terrier | 9 |
| Russian Blue | Bombay | 9 |
| Basset Hound | Beagle | 9 |
| Maine Coon | Bengal | 8 |
| American Pit Bull Terrier | German Shorthaired | 8 |

报告中可重点说明：错误主要集中在外观、毛色、体型或面部结构相近的品种之间，例如斗牛犬相关类别、长毛猫类别和灰/黑色短毛猫类别。

## 分类失败案例分析要点

`fig_failure_cases.png` 选取 EfficientNet-B0 的高置信错误样例。图中可见部分错误并非随机误判，而是来自视觉特征相似或图像质量因素：

- Russian Blue 被预测为 Bombay：二者在深色短毛、低纹理图像中容易混淆。
- Boxer 被预测为 American Bulldog：脸部轮廓、短毛和白色体表区域相近。
- Maine Coon 被预测为 Bengal：局部裁剪或纹理特征可能强化了斑纹线索。
- Basset Hound 被预测为 Beagle：垂耳、棕白毛色和正面姿态相近。
- American Pit Bull Terrier 被预测为 Staffordshire Bull Terrier：类别本身外观差异较细。

## 分割预测图分析要点

`fig_seg_predictions.png` 上半部分为测试集中高 mIoU 样例，下半部分为低 mIoU 困难样例。分割测试集总体结果为 Pixel Accuracy 0.961981、mIoU 0.924921、Dice 0.954638。

当前困难样例选择规则为：优先从目标 mask 同时包含前景和背景的样本中选取最低 mIoU 样例，避免把全背景标注异常样本作为主要失败案例。

低 mIoU 样例包括：

| dataset index | mIoU | Dice | 说明 |
| ---: | ---: | ---: | --- |
| 2005 | 0.0189 | 0.0728 | 近景猫脸与真实 mask 区域差异明显，边界和主体范围判断失败。 |
| 2302 | 0.0473 | 0.1729 | 白色宠物与浅色背景、裁剪区域共同导致前景范围偏差。 |
| 2308 | 0.0815 | 0.2802 | 长毛白色宠物在草地背景下边界复杂，预测区域与真实 mask 不一致。 |

报告中可将分割失败原因归纳为：复杂毛发边界、浅色主体与背景对比不足、近景裁剪导致上下文不足，以及 trimap 标注边界与模型预测边界不完全一致。

