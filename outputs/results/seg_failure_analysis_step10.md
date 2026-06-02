# Step 10 segmentation failure analysis

Visualization source: `D:\DeepLearning Workplace\Project\05-CV Assignment\outputs\figures\fig_seg_predictions.png`.

The global test metrics are high, and the high-mIoU examples show that the model generally captures the pet foreground rather than collapsing to all-background or all-foreground predictions.

The lowest-mIoU samples indicate failure modes that should be discussed in the report:
- Dataset index 2292: mIoU=0.0000, Dice=0.0000. The original trimap contains only value 2, so the binary ground truth becomes all background; this should be described as an annotation/data-quality edge case.
- Dataset index 1858: mIoU=0.0081, Dice=0.0000. The original trimap also contains only value 2, producing an all-background target; this should be excluded from claims about ordinary boundary quality and discussed as a label anomaly.
- Dataset index 2005: mIoU=0.0189, Dice=0.0728. The trimap contains ordinary 1/2/3 values, but the prediction covers a largely different foreground region, making it a genuine hard segmentation failure case.

For the final report, the failure discussion should distinguish dataset annotation anomalies from genuine model errors. Genuine model errors can be grouped under boundary ambiguity, complex background, unusual crop/scale, or foreground-background confusion after visual inspection.
