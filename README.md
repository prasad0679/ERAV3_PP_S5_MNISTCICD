[![ML Pipeline](https://github.com/prasad0679/ERAV3_PP_S5_MNISTCICD/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/prasad0679/ERAV3_PP_S5_MNISTCICD/actions/workflows/ml-pipeline.yml)

# MNIST DNN Classifier 🚀⭐

A simple CNN-based classifier for MNIST digits with CI/CD pipeline.

## Objective: Make a MNIST based model that has following characteristics:

Has less than 25000 parameters
* Gets to training accuracy of 95% or more in 1 Epoch
* Once done, upload to GitHub "with actions". On actions you should be testing for:
  * Model has less than 25000 parameters
  * Has accuracy of more than 95% in 1 Epoch
* Add image augmentation to your code, and share screenshot of augmented image samples
* Add 3 more unique and relevant tests and share python code for these tests (these tests should be passing in Actions)
* Add "build pass" Badge to your README.md

## MNIST Model Architecture and Training Summary:

![ModelArch TrainingSummary](https://github.com/user-attachments/assets/3f489fe9-9f50-4300-9c2a-52a4777a57fd)

## Python Tests Conducted
1. Test Number of Model parameters: Should be < 25000 parameters
2. Test input image shape: check that the model takes input image of 28X28 and return the 10 outputs
3. Test Model Accuracy after 1 epoch: Should be > 95%
4. Test if model outputs valid probability distributions
   * Verifies softmax outputs sum to 1
   * Ensures probabilities are between 0 and 1
6. Test if model can handle different batch sizes :
   * [1, 32, 64, 128] : Default batch size used is 512
   * Ensures correct output shape for each batch size
8. Test if model gradients are properly computed
   • Checks if gradients exist and are non-zero
   * Ensures the model can learn through backpropagation

## Applied Image Augmentations.
• Random rotation (±10 degrees)
* Random translation (±10% in both directions)

![ImageAugmentationCode](https://github.com/user-attachments/assets/6f63b16d-f658-43a5-bb2b-3534fb64bf72)

## Image Augmentation Samples: 

![ImageAugmentationSamples](https://github.com/user-attachments/assets/e4cf642e-9539-49d6-9aef-bf221e0a400d)




