# Fire Detection system using Computer Vision 

## Overview 
Accidental fires are one of the biggest challenges in disaster control. They are highly uncertain and can occur in forests, workplace, industries and households to name a few. Uncontrolled fires can quickly turn into massive conflagrations, and cause immense damage to biological life, goods, establishments and the environment. This places an importance on the development of efficient fire detection systems that can quickly identify fires as soon as it starts, especially in remote locations such as forests and grasslands. Conventional detection systems mainly rely on temperature or smoke to physically reach the sensors in order to invoke a response may delay the process of identifying the outbreak of an uncontrolled fire. Advancements in the field of computer vision at breakneck speeds over the last few decades have given rise to CNN architectures that can visually identify objects efficiently, these concepts can be used to detect smoke in a fraction of the time taken by its conventional fire detection counterparts. Liu and Ahuja, 2004 and Enis Çetin et al., 2013 propose two such methodologies. Arson and the lack of education about fire emergency protocols also contribute to the occurrence and poor management of accidental fires.

## Dataset 
The dataset was obtained using a custom webscraping tool, the code for the webscraper can be found in the script _Imagescraper.py_ can be found under the Data folder in this repository. The images used to create the dataset were captured using low resolution CCTV cameras. The dataset contains three classes, "Smoke", "Fire" and "Normal", a few samples of the dataset can be found below.

<img width="1297" alt="Screenshot 2021-10-24 at 2 14 15 PM" src="https://user-images.githubusercontent.com/69090777/138586835-f6a42b30-311e-4be4-84e7-976cd74825ae.png">

## Models 
This project experimented with three Models, ResNet50, MobileNetV2 and FireNet. The implementation fo FireNet implemented by _Jadon et al.(2019)_. 
Initial parameters for all three models can be found here :  https://drive.google.com/drive/folders/1NbijGD-a0WyWLnf9x21m76NwLLWpqnQk?usp=sharing

## Results 
A comparision of the accuracies of the three models tested are below.

<img width="676" alt="Screenshot 2021-10-24 at 2 27 36 PM" src="https://user-images.githubusercontent.com/69090777/138587256-36d873aa-6fce-4f97-81cd-8f8e50caa1e0.png">

ResNet50 outperformed both FireNet and MobileNetV2 as expected. However, it is surprising that FireNet was able to classify images with a higher accuracy when compared to MobileNetV2. 

## Hardware 
All three models were deployed and tested on a RaspberryPi model 4b with an additional camera module. 
Below is the picture of the hardware setup. 

<img width="750" alt="image" src="https://user-images.githubusercontent.com/69090777/138587359-66404154-3e23-4cfa-b705-9a4bc424e3d0.png">


## References 
[1] C.B. Liu. and N. Ahuja. 2004, Vision based fire detection. 

[2] A. Enis Çetin., K. Dimitropoulos., B. Gouverneur., N. Grammalidis., O. Günay., Y. H. Habiboglu., B. U. Töreyin. and S. Verstockt. 2013, Video fire detection.

[3] P. Li., and W. Zhao. 2020, Image fire detection algorithms based on convolutional neural networks. 

[4] X.Zhang., X. Zhou. and M. Lin, J. Sun. CVPR 2018, ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices 

[5] K. He., X. Zhang., S. Ren. and J. Sun. 2015, Deep Residual Learning for Image Recognition.

[6]L. Yandong., H. Zongbo. and L. Hang. 2016, Survey of Convolutional Neural Networks. 

[7]W. Gomaa. 2021, Deep Architectures in Visual Transfer Learning. 

[8]Z. Qin., Z. Zhang., S. Zhang., H. Yu., J. Li., and Y. Peng. IJCNN 2018, Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications. 

[9]A.Jadon., M. Omama., A. Varshney., M. Ansari., R. Sharma. 2019, FireNet: A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications.

 _*This repository contains all the code used for my 6th sememster project under Prof. Sowmyashree S, department of Electronics and Telecommunications Engineering of BMSIT&M 
This project uses Python for webscraping to generate the dataset and MATLAB to train the the models for the object detection task which are then deployed on raspberryPi._ 
