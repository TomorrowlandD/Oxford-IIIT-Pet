# Step 10 segmentation failure analysis

Visualization source: `D:\DeepLearning Workplace\Project\05-CV Assignment\outputs\figures\fig_seg_predictions.png`.

The global test metrics are high, and the high-mIoU examples show that the model generally captures the pet foreground rather than collapsing to all-background or all-foreground predictions.

The lowest-mIoU samples are selected from targets containing both foreground and background where possible. They indicate failure modes that should be discussed in the report:
- Dataset index 2005: mIoU=0.0189, Dice=0.0728. This sample should be treated as a hard case for boundary/foreground localization and checked visually in the generated comparison figure.
- Dataset index 2302: mIoU=0.0473, Dice=0.1729. This sample should be treated as a hard case for boundary/foreground localization and checked visually in the generated comparison figure.
- Dataset index 2308: mIoU=0.0815, Dice=0.2802. This sample should be treated as a hard case for boundary/foreground localization and checked visually in the generated comparison figure.

For the final report, these cases can be grouped under boundary ambiguity, complex background, unusual crop/scale, or foreground-background confusion after visual inspection.
