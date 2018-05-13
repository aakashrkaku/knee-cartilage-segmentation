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


