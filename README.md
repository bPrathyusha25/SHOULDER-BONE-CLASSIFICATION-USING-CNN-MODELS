# SHOULDER-BONE-CLASSIFICATION-USING-CNN-MODELS
SHOULDER BONE CLASSIFICATION USING CNN MODELS

For different causes, fractures develop in the shoulder region, which has a more excellent range of motion than other joints in the body. Computed tomography (CT) or magnetic resonance imaging (MRI) are used to identify these fractures, whereas X-ray is utilised to examine the bones. The purpose of this research is to aid clinicians by using artificial intelligence to categorise shoulder pictures obtained from X-ray machines as fracture/non-fracture. In this report, seven deep convolution neural network-based models are trained and built to identify the shoulder fractures using a modified version of the extensive MURA dataset that includes pictures of the shoulders. A random split was done so that the testing and training sets included 563 and 8379 photos, respectively. These testing and training set consist of positive (Fracture) and negative (non - fracture). Of the 563 picture sets to be tested, there are 285 non-fracture pictures and 278 fracture photos. All the models are trained and compared using various performance measures, and the best model is selected. The ensemble model has bested all the other models with a training accuracy of 86.25% and 77.3% validation accuracy.



Project Objectives 

Produce the pre-processing steps performed on the MURA dataset. 
The necessary pre-processing steps include cropping the images, converting RGB images into greyscale, and performing rotations wherever necessary. 
Perform experiments with different deep learning convolutional neural network models and discuss their prediction performance and accuracy. 
Complex deep CNN models will be dealt with in an industry-leading manner while performing a literature evaluation during the project's study and development stage. 
This workflow then follows the dataset through to completion and where the outcomes are then reported and discussed. 
When the best performing strategy has been discovered, additional analysis will be done to find out the best. 
Discuss model selection by providing advice concerning the choice of model based on the use case of shoulder fracture predictions. 
A final model will be proposed based on the performance and application of shoulder bone X-ray classification.
The literature review findings and the experiment findings will be combined to provide an amalgamation of the results.



DataSet :

The dataset was taken from Kaggle
Link: https://www.kaggle.com/datasets/cjinny/mura-v11
The MURA dataset, a vast collection of positive and negative bone X-ray scans, was used in this study. This dataset was open to the public for a competition where contestants competed to test if the models could perform with the same level of accuracy as professional radiologists. MURA is a musculoskeletal radiograph dataset comprising 14,863 investigations from 12,173 individuals, totalling 40,561 multi-view radiographic pictures. Each corresponds to one of seven different upper 𝑘 = po − pe 1 − pe 31 extremity radiography studies: shoulder, humerus, finger, elbow, wrist, forearm, and hand. The dataset is obtained from Kaggle, which is publicly available. However, in this report, only a part of the dataset is used, consisting of the shoulder region images. This modified dataset consists of 8942 images of two classes, positive and negative. The positive class consists of 4446 photos, and the negative class consists of 4496 images.


Prerequisites:

Python 
TensorFlow 
pandas 
numpy 
tensorflow 
open cv
pickle

A)	Pre-processing of the sample image :  
Data Generator class have been used to split the data for training and testing set randomly. 
First, the data Generator takes all the image filenames from data and their respective labels as input. 
Then, the transformation is applied and returns the data into a feature target format, i.e., transformed image and its label as the output. 
The images are first normalised and resized into 300 × 300 shaped images and cropped from the centre in 224x224 size. 
Then the images are shuffled and split into training and testing dataset. 
Thus, the training data has 8379 images where 4168 images are for class 0 (positive), and 4211 images are for class 1 (negative). 
The testing data has 563 images, where 278 images are for class 0 and 285 images for class 1.


B)	Later 3 different models are trained  
All the models used here follows the same pattern, which is as follows: 

• The base model is among pre-trained models with weights used from the model trained on the ImageNet dataset and removal of top layers as we would be adding our designed layers to it. 33
 • Next, the output from the base model is flattened and passed on to the next layer, which is a dense layer with 256 neurons. 
• The output of this dense layer is passed on to the activation layer to add non-linearity to the system. 
• A dropout of 50% is applied to the layer. 
• The Next layer is the final layer with a dense layer with two neurons and a softmax activation function. 
• The dependent variable is categorical, 'categorical cross entropy' is selected as loss function and metrics as 'accuracy''. 
• Finally, all the models are trained using five epochs. and  the accuracy of the model is tested by calculating the below performance metrics
1.	Accuracy
2.	Recall / Sensitivity
3.	F1 Score
4.	Area Under Curve (AUC)
5.	Cohen's Kappa score

