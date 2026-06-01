# AGENTS.md

## 项目定位

本项目是学生课程实验报告项目，主题为 **Oxford-IIIT Pet 数据集上的宠物品种分类与前景语义分割综合实践**。

核心实验范围保持收敛：
- 分类任务：37 类宠物品种分类，使用 `ResNet18` 作为 baseline，`EfficientNet-B0` 作为主模型。
- 分割任务：宠物前景二分类语义分割，使用 `DeepLabV3-MobileNetV3` 作为主模型。
- 轻量消融：优先复用 `ResNet18 vs EfficientNet-B0` 对比；`U-Net`、`Grad-CAM`、分割辅助分类均为可选扩展，不作为主线。

## 环境与运行

- 项目 Python 环境固定使用 conda 环境 `unet`，路径为 `D:\Anaconda3\envs\unet`。
- 优先使用该环境下的 Python 执行脚本，例如：
  ```powershell
  D:\Anaconda3\envs\unet\python.exe src\train_cls.py
  ```
- 深度学习实现以 PyTorch / TorchVision 为主，不随意引入额外框架。
- 随机划分、训练和评估代码应设置固定随机种子，保证结果可复现。

## 报告与 LaTeX

- 实验报告以现有 `main.tex` 模板为准，不随意重写模板结构和字体设置。
- 全程使用 `xelatex` 编译报告，例如：
  ```powershell
  xelatex main.tex
  ```
- 报告内容应围绕数据处理、模型训练、指标评估、可视化和结果分析展开。
- 不虚构实验指标；结果表格中的 Accuracy、Macro-F1、mIoU、Dice 等必须来自真实运行输出。

## 建议目录约定

- `data/`：数据集文件。
- `src/`：训练、评估、模型、数据集和工具代码。
- `outputs/checkpoints/`：模型权重。
- `outputs/logs/`：训练日志。
- `outputs/figures/`：训练曲线、混淆矩阵、mask 可视化等报告图片。
- `outputs/results/`：指标 JSON 或 CSV。

## 文件操作约束

禁止批量删除文件或目录。

不要使用：
- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件，例如：
```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量删除文件，应停止操作，并请求用户手动删除。
