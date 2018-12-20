# Leukocyte-classification
Using deep learning (in keras) to classify leukocytes in the 4 most common classes (Eosinophil, Lymphocyte, Monocyte, Neutrophil). Basophils were excluded for being much less frequent.
Originally based on [this work](https://github.com/dhruvp/wbc-classification), whose main objective is a binary classification. In my work I'm trying a multi-class classification.

## Purpose
Find the best architectures to classify 4 categories of leukocytes.

Note: Thesis and PDF(Powerpoint) Presentation in portuguese.

--------------------------------------------------------- 
# Obtaining Deep Learning Models for Automatic Classification of Leukocytes

The number of leukocytes present in the blood provides relevant information regarding the state of the immune system, allowing the assessment of potential health risks. These cell bodies are usually classified into 5 categories: lymphocytes, monocytes, neutrophils, eosinophils and basophils based on morphological and physiological characteristics.
This work describes the classification of leukocyte images using the winning neural network architectures of the annual ILSVRC software contest. 
The classification of leukocytes is made using pre-trained networks and the same models trained from scratch, in order to select the ones that achieve the best performance for the intended task. The leukocytes categories used were eosinophils, lymphocytes, monocytes and neutrophils. Possibly due to the low prevalence in the blood, it is difficult to obtain enough images of basophils and band leukocytes, so they were excluded from the study.

## Languajes, Frameworks and Implementation

The Jupyter Notebook, with the Python language and the Keras API (using the TensorFlow as the backend) were used for the construction and implementation of the leukocyte classification solution.
The neural architectures used with the Keras library were the ResNet50, the VGG16, the VGG19, the InceptionV3, the Xception, the Densenet121, the Densenet169 and the Densenet201; which are based on articles from reference networks such as LeNet-5, AlexNet, GoogLeNet / Inception, VGGNet and ResNet.
All the models used were trained in two different ways: using transfer learning and trained from scratch. Transfer learning consists of transferring the values of the weights acquired in a previous training (in this case with 1000 categories from the original training in the ImageNet dataset) and it is required to train only the top of the network to recognize specific characteristics of the dataset in question, and to define the number of categories to be classified.
Training from scratch involves training the entire network, which is usually initialized randomly and is a, significantly, more extended process. 

## The dataset

The dataset used in training is usually preprocessed and, in the case of this work, consisted on the selection, classification and segmentation of images containing eosinophils, lymphocytes, monocytes and neutrophils. This preprocessing stage is also prone to human error, but it is the starting point for the automated classification performed by the neural networks.
The data set used initially in the development of the project contains 598 leukocyte images. Most of the images (approximately 352) come from a repository on GitHub. The remaining images were obtained through a Google search according to specific terminology, namely 'monocyte', 'leucocyte', 'lymphocyte' and 'neutrophil'. The dimensions of the dataset images vary significantly according to their provenance. 
A part of the development of this work was spent on the preprocessing of the dataset. It was necessary to label each of the collected images according to the 4 categories to be classified. Subsequently, each image was manually segmented in order to provide the neural network with a register, as specific as possible, of the cell pixels whose classification was intended.
In the first stage of preprocessing the initial set of images was segmented manually using GIMP2, which allowed to zoom in and centre the cells in the images. The data augmentation was then performed, resulting in a total of 10466 images distributed in the 4 classes, each having approximately 2600 training images, 215 validation images and 215 test images.
However, this first stage of preprocessing did not work as expected, and the models were unable to train properly, so it was necessary to review all available data and proceed to a newly refined segmentation of the images.
Starting from the initial set of images, a second selection and segmentation procedure was then performed, from which 100 training images, 10 validation images and 10 test images were obtained for each class. When manual segmentation was done PAINT3D was used to zoom in and remove the background of each of the images, preserving only the foreground for each cell to be classified.
After a new data augmentation process, 4394 training images, 440 validation images and 440 test images were distributed uniformly across the four categories.

## Results
The training phase was performed during 100 epochs for the models trained from scratch and for the transfer learning ones.
In the evaluation stage, it was found that, despite the high accuracy and low loss results of the models trained from scratch, they were overfitting and did not achieve good test scores. The networks using transfer learning achieved, on average, classification accuracies of 87.8% against 26.3% of the models trained from scratch. 
The best performing networks were the DenseNet169 and the DenseNet201 both with 98.2% accuracy. Moreover, the average time spent training the models by transfer learning was 12 times less than the time taken by the models trained from scratch.
Since it was intended to compare the performance of the models with and without transfer learning, no changes were made to the settings from one group of models to the other one. The solution to improving the results in these models would be based on the use of regularization, which introduces slight modifications in the learning algorithm in order to allow a better generalization capability, which translates into an improvement in network performance at the test stage. Some common regularization techniques are L2 and L1 regularization, dropout and early stopping.
In the attempt to improve the performance of the models with transfer learning, the training of the models was carried out for 5000 epochs. The results, on the test group, showed an average increase of 1.68% in the percentage of accuracy of these networks.
The best results obtained with some deep learning models are directly related to the preprocessing work performed on the images. The absence of background in the images allowed the networks a more natural extraction of the pertinent features to the classification process. During the first preprocessing phase, in the segmentation process, where only a zoomed area containing the cell to be classified was selected (leaving some background), the classification results in the test dataset were significantly lower than those obtained after the background removing from the leukocytes images. In this sense, it is understood the importance of data processing before providing it to neural networks for training (as well as for the test phase).

## Conclusion

The analysis of the results obtained from this work should be done considering the number of images used, as well as the uncertainty brought by the manual classification. The use of a data set composed of more cell images (with an increase of two or three orders of magnitude) would be significant as a way of corroborating with greater confidence the results obtained and would undoubtedly allow the development of the project with more practical applicability as only 4 categories were used.
Although there is some variability in the morphology of the images, illumination and microscopy artefacts, the selected approaches caused that those different conditions do not weight too much on the final result and it can be affirmed that the existence of images from different sources even constituted a way of increasing the generalization power of the networks.
It is probably possible to attain better results with more recent network architectures or with more recent versions of the models used in this project, but the results attained are already outstanding.
With this work, it was possible to implement a set of networks for the classification of 4 categories of leukocytes with percentages above 98% in the available data set. It is highly probable that with a more extensive set of images and the coupling of a computational mechanism of identification and segmentation of images (and this could also be done by a neural network) it is possible to automate the identification and the counting of all the existing types of leukocytes from microscopy images.
Although the methodology presented in this work is relatively simple, it can be considered as an initial step for the development of a more complex device with greater autonomy in the task of classifying leukocytes and blood cells in general.


