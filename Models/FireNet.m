% Here is the MATLAB code to train the FireNet for Fire and Smoke Detection.
% Load Parameters
% Find Initial Parameters here : https://drive.google.com/drive/folders/1NbijGD-a0WyWLnf9x21m76NwLLWpqnQk?usp=sharing 
trainingSetup = load("path to file containing initial training parameters");

% Data Import
imdsTrain = imageDatastore("path to dataset","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.7,"randomized");

% Data Augmentation
imageAugmenter = imageDataAugmenter(...
    "RandRotation",[10 10],...
    "RandXReflection",true,...
    "RandYReflection",true);
augimdsTrain = augmentedImageDatastore([227 227 3],imdsTrain,"DataAugmentation",imageAugmenter);
augimdsValidation = augmentedImageDatastore([227 227 3],imdsValidation);

% Training Options
opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",10,...
    "MiniBatchSize",120,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",25,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

% Layer Array
layers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
    averagePooling2dLayer([5 5],"Name","avgpool2d_1","Padding","same")
    dropoutLayer(0.5,"Name","dropout_1")
    convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
    averagePooling2dLayer([5 5],"Name","avgpool2d_2","Padding","same")
    dropoutLayer(0.5,"Name","dropout_2")
    convolution2dLayer([3 3],32,"Name","conv_3","Padding","same")
    averagePooling2dLayer([5 5],"Name","avgpool2d_3","Padding","same")
    dropoutLayer(0.5,"Name","dropout_3")
    fullyConnectedLayer(10,"Name","fc_1")
    dropoutLayer(0.5,"Name","dropout_4")
    fullyConnectedLayer(10,"Name","fc_2")
    fullyConnectedLayer(3,"Name","fc_3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%Train Network
[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);

