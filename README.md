# Intel-images-classification
A comparison between Transfer Learning and custom Convolutionnal Network to classify images.

#### TS;WR (Too short; Will Read)
Read the full report (in french) here: [Report.pdf](https://github.com/BecayeSoft/Intel-images-classification/blob/main/Report.pdf)

## Situation
Transfer learning is used to solve image classification problems when we don't have time or computational power.
But does it work all the time?

Here, I compare EfficientNet performances and a custom Convolutional Neural Network to classify images. 

We have 5 classes:
* Mountain
* Sea
* Building
* Forest
* Iceberg

I used this dataset from kaggle: [Intel-image](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

## Action
I built two models:
* Custom CNN: a simple CNN
* EfficientNet: a state-of-the-art model for image classification wiht ImageNet weights.

## Results
EfficientNet performed poorly, with an accuracy of less than 20% while my custom CNN yielded a result of 83%.

Note that, the dataset is not good quality and some images are mislabeled, which I believe, affected my model's performances.

#### Custom CNN

<img width="670" alt="image" src="https://user-images.githubusercontent.com/87549214/232185130-a9dd15bb-2a57-40c2-8002-97a6217fc0df.png">

#### EfficientNet

<img width="696" alt="image" src="https://user-images.githubusercontent.com/87549214/232185158-29f6c265-beac-46e2-a6d8-4673ddfe2e58.png">
