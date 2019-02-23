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
    tempX = intermediate{2, :, i};
    tempY = intermediate{3, :, i};
    XTrain{i} = [];
    XTrain{i} = tempX(:,1:end-1);
    YTrain{i} = [];
    YTrain{i} = tempY(1:2,2:end);
end

intermediate = struct2cell(reshape(testData, 1, numTest * 8));

XTest = cell(size(intermediate, 3), 1);
YTest = cell(size(intermediate, 3), 1);

for i = 1:size(intermediate, 3)
    tempX = intermediate{2, :, i};
    tempY = intermediate{3, :, i};
    XTest{i} = [];
    XTest{i} = tempX(:,1:end-1);
    YTest{i} = [];
    YTest{i} = tempY(1:2,2:end);
end

%%  

inDim = 98;
hiddenUnits1 = 100;
hiddenUnits2 = 75;
fullyConnected = 50;
dropout = 0.1;
outDim = 2;

layers = [ ...
    sequenceInputLayer(inDim)
    lstmLayer(hiddenUnits1,'OutputMode','sequence')
   % lstmLayer(hiddenUnits2,'OutputMode','sequence')
    fullyConnectedLayer(fullyConnected)
    dropoutLayer(dropout)
    fullyConnectedLayer(outDim)
    regressionLayer];

maxEpochs = 200;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold', 1, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'LearnRateDropPeriod',150, ...
    'LearnRateDropFactor',0.1, ...
    'Plots','training-progress',...
    'Verbose',0);

net = trainNetwork(XTrain, YTrain, layers, options);

YPred = predict(net, XTest, 'MiniBatchSize', 1);

%%
figure
plot(YTest{1}(1,:), YTest{1}(2,:))
hold on
plot(YPred{1}(1,:),YPred{1}(2,:))
legend('Actual', 'Predicted')

figure
plot(YTest{2}(1,:), YTest{2}(2,:))
hold on
plot(YPred{2}(1,:),YPred{2}(2,:))
legend('Actual', 'Predicted')