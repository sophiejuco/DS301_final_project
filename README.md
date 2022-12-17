# DS301_final_project
DS-301 Final Project: Document localization with transfer learning (al6253 &amp; smj490)

## Regression head training only
Time: ~639s/step

Average IoC: 0.370

Prediction shape errors: 0 (out of 6158)

![alt text](https://github.com/sophiejuco/DS301_final_project/blob/main/model0sts.png?raw=true)


## Partial fine-tuning (8 layers)
Time: ~678s/step

Average IoC: 0.693

Predicition shape errors: 3 (out of 6158)

## Full fine-tuning
Time: ~749s/step

Average IoC: 0.822

Prediction shape errors: 0 (out of 6158)
