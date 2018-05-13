# knee-cartilage-segmentation
The project covers various deep learning and traditional machine learning models to automate the segmentation of knee cartilages using the diffusion weighted MRIs.

The best deep learning model performs at the human-expert level. 

Below are the results for the different Deep Learning Models.

| Model Name                                      | Dice Score (Femur) | Dice Score (Patella) | Dice Score (Tibia) |
|-------------------------------------------------|--------------------|----------------------|--------------------|
| Baseline U-Net Model                            | 0.671              | 0.745                | 0.573              |
| Dilated U-Net Model                             | 0.681              | 0.764                | 0.580              |
| Dilated U-Net (L2 Regularized)                  | 0.683              | 0.751                | 0.552              |
| Small U-Net                                     | 0.678              | 0.773                | 0.593              |
| Small Dilated U-Net                             | 0.670              | 0.771                | 0.621              |
| Ensemble of Small U-Net and Small Dilated U-Net | 0.689              | 0.783                | 0.640              |
| Human Expert Re-segmentation Dice Score         | 0.711              | 0.743                | 0.629              |

# Visualizartion of the predictions and the ground truth
1. Case where  the model correctly predicted ground truth. Left: Original Image, Center: Ground Truth, Right: Model Prediction. Femur = Pink, Patella = Red, Tibia = Yellow
![](https://github.com/aakashrkaku/knee-cartilage-segmentation/blob/master/plots/pred_2.png)
2. Case where the model correctly predicted a segment not present in the ground truth. Left: Original Image, Center: Ground Truth, Right: Model Prediction. Femur = Pink, Patella = Red, Tibia = Yellow
![](https://github.com/aakashrkaku/knee-cartilage-segmentation/blob/master/plots/pred_1.png)
3. Case where the model was more conservative than the human expert. Left: Original Image, Center: Ground Truth, Right: Model Prediction. Femur = Pink, Patella = Red, Tibia = Yellow
![](https://github.com/aakashrkaku/knee-cartilage-segmentation/blob/master/plots/pred_4.png)
4. Case where the human expert was more conservative than the model. Left: Original Image, Center: Ground Truth, Right: Model Prediction. Femur = Pink, Patella = Red, Tibia = Yellow
![](https://github.com/aakashrkaku/knee-cartilage-segmentation/blob/master/plots/pred_3.png)

# Confidence maps
Confidence maps are made that helps the user to understand how confident the model is of the predictions.

A confidence map for two example validation images can be seen below. The circled stray voxels are incorrectly classified as one of the cartilage. It can be seen that incorrect voxels have low confidence.
![](https://github.com/aakashrkaku/knee-cartilage-segmentation/blob/master/plots/cert_image.png)



