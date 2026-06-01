# Oxford-IIIT Pet 分类与前景分割实验

## 项目目录结构

```text
data/
src/
  config.py
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
