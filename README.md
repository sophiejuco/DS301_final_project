# DS301_final_project
DS-301 Final Project: Document localization with transfer learning (al6253 &amp; smj490)
<br>
This repository contains the code notebooks used to experiment with using a pre-trained ResNet-18 to recognize document corners in video frames.<br>
* <em>csv-cleaning.ipynb</em> was used to remove unnecessary characters from the ground truth files.
* <em>GDrive_to_GCS.ipynb</em> was used to transfer the dataset from Google Drive to Google Cloud Storage.
* <em>training.ipynb</em> was used to load and train the CNN.
* <em>prediction-analysis.ipynb</em> was used to evaluate model performance using the intersection-over-union (IoU) metric.
* <em>utils.py</em> contains helper functions for analysis and visualization.
<br>

There are also some folders:<br>
* <em>Prediction plots</em> contains sample results from our trained models.
* <em>Training plots</em> contains graphs of training metrics.
<br>
The dataset used can be downloaded from this Google Drive folder: https://drive.google.com/drive/folders/1N9M8dHIMt6sQdoqZ8Y66EJVQSaBTq9cX?usp=share_link.
<br>

# Results:
## Regression head training only
Time: ~639s/step

Average IoU: 0.370

Prediction shape errors: 0 (out of 6158)

![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/Training-plots/model0sts.png?raw=true)

Example - test image 0: IoU = 0.207
![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/Prediction-plots/pred0.png?raw=true)

## Partial fine-tuning (8 layers)
Time: ~678s/step

Average IoU: 0.693

Predicition shape errors: 3 (out of 6158)

![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/Training-plots/pft_model_sts.png?raw=true)

Example - test image 0: IoU = 0.723
![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/Prediction-plots/pred1.png?raw=true)

## Full fine-tuning
Time: ~749s/step

Average IoU: 0.822

Prediction shape errors: 0 (out of 6158)

![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/Training-plots/fft_model_sts.png?raw=true)

Example - test image 0: IoU = 0.814
![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/Prediction-plots/pred2.png?raw=true)
