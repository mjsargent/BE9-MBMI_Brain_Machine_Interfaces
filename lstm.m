clear all; close all
clc

%% prepare training data
load('monkeydata_training.mat');

% set random number generator
rng(2013);

% shuffle the data
ix = randperm(length(trial));

% split into train and test sets
trainTestSplit = 0.8;
numTrain = trainTestSplit * length(trial);
numTest = length(trial)-trainTestSplit * length(trial);

trainingData = trial(ix(1:numTrain),:);
testData = trial(ix(numTrain+1:end),:);

intermediate = struct2cell(reshape(trainingData, 1, numTrain * 8));

XTrain = cell(size(intermediate, 3), 1);
YTrain = cell(size(intermediate, 3), 1);

for i = 1:size(intermediate, 3)
    XTrain{i} = intermediate{2, :, i};
    YTrain{i} = intermediate{3, :, i};
end

intermediate = struct2cell(reshape(testData, 1, numTest * 8));

XTest = cell(size(intermediate, 3), 1);
YTest = cell(size(intermediate, 3), 1);

for i = 1:size(intermediate, 3)
    XTest{i} = intermediate{2, :, i};
    YTest{i} = intermediate{3, :, i};
end

%%  

inDim = 98;
hiddenUnits = 100;
fullyConnected = 50;
dropout = 0.2;
outDim = 3;

layers = [ ...
    sequenceInputLayer(inDim)
    lstmLayer(hiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(fullyConnected)
    dropoutLayer(dropout)
    fullyConnectedLayer(outDim)
    regressionLayer];

maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

net = trainNetwork(XTrain, YTrain, layers, options);

YPred = predict(net, XTest, 'MiniBatchSize', 1);
