%% Load data
clear all; close all;
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


%% Define parameters
% global
eta = 0.01;
inputSize = 98;
% LSTM parameters
hiddenLayers = 100;
concatLayers = inputSize + hiddenLayers;

forgetWeight = 0.001*randn(hiddenLayers, concatLayers);
forgetBias = 0.001*randn(hiddenLayers,1);

inputWeight = 0.001*randn(hiddenLayers, concatLayers);
inputBias = 0.001*randn(hiddenLayers, 1);

cellUpdateWeight = 0.001*randn(hiddenLayers, concatLayers);
cellUpdateBias = 0.001*randn(hiddenLayers,1);

outputWeight = 0.001*randn(hiddenLayers, concatLayers);
outputBias = 0.001*randn(hiddenLayers,1);

predictWeight = 0.001*randn(2, hiddenLayers);
predictBias = 0.001*randn(2, 1);

LSTM = struct;

LSTM.inputSize = inputSize;
LSTM.hiddenSize = hiddenLayers;

LSTM.forgetWeight = forgetWeight;                           % Weight
LSTM.forgetWeightD = zeros(hiddenLayers, concatLayers);     % Gradient
LSTM.forgetWeightM = zeros(hiddenLayers, concatLayers);     % AdaGrad Momentum

LSTM.forgetBias = forgetBias;
LSTM.forgetBiasD = zeros(hiddenLayers,1);
LSTM.forgetBiasM = zeros(hiddenLayers,1);

LSTM.inputWeight = inputWeight;
LSTM.inputWeightD = zeros(hiddenLayers, concatLayers);
LSTM.inputWeightM = zeros(hiddenLayers, concatLayers);

LSTM.inputBias = inputBias;
LSTM.inputBiasD = zeros(hiddenLayers, 1);
LSTM.inputBiasM = zeros(hiddenLayers, 1);

LSTM.cellUpdateWeight =  cellUpdateWeight;
LSTM.cellUpdateWeightD = zeros(hiddenLayers, concatLayers);
LSTM.cellUpdateWeightM = zeros(hiddenLayers, concatLayers);

LSTM.cellUpdateBias =  cellUpdateBias;
LSTM.cellUpdateBiasD = zeros(hiddenLayers, 1);
LSTM.cellUpdateBiasM = zeros(hiddenLayers, 1);

LSTM.outputWeight =  outputWeight;
LSTM.outputWeightD = zeros(hiddenLayers, concatLayers);
LSTM.outputWeightM = zeros(hiddenLayers, concatLayers);

LSTM.outputBias = outputBias;
LSTM.outputBiasD = zeros(hiddenLayers, 1);
LSTM.outputBiasM = zeros(hiddenLayers, 1);

LSTM.predictWeight = predictWeight;
LSTM.predictWeightD = zeros(2, hiddenLayers);
LSTM.predictWeightM = zeros(2, hiddenLayers);

LSTM.predictBias =  predictBias;
LSTM.predictBiasD = zeros(2,1);
LSTM.predictBiasM = zeros(2,1);

%% Training Loop - batch size 1 (SGD, Adagrad)
maxEpochs = 50;
totalLoss = zeros(1,maxEpochs);
for epoch = 1:maxEpochs
    for trial = 1:length(XTrain)
        % Set t-1 LSTM memory states
        hidden_init = zeros(LSTM.hiddenSize,1);
        C_init = zeros(LSTM.hiddenSize,1);
        % Extract info from cells
        XTrial = XTrain{trial};
        YTrial = YTrain{trial};

        [loss,predictStore, hidden, C, LSTM] = completePass(XTrial, YTrial, hidden_init, C_init, LSTM);
        totalLoss(:,epoch) = 0.5 * (sum(loss).^0.5);
        
        LSTM = step(LSTM, eta);
        disp("Epoch:")
        disp(epoch)
        disp("Trial:")
        disp(trial)
    end
end
%TEST = XTrain{1};
%[z, forget, input, C_bar, C,output, hidden, TESTOUT] = forward(TEST(:,1), randn(LSTM.hiddenSize,1), randn(LSTM.hiddenSize, 1), LSTM); 
%% FUNCTION DEFINITIONS %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define activation functions

% sigmoid
function output = sig(input)
output = 1./(1 + exp(-input));
end

% tanh - built in

%% Define derivative functions for backward pass

% derivative of sigmoid - input is sig(X)
function output = dsig(input)
output = input.*(1-input);
end

% derivative of tanh - input is tanh(X)
function output = dtanh(input)
output = 1 - (input .* input);
end

%% Define forward pass

function [z,forget,input,C_bar, C,output,hidden,predict] = ...
    forward(x, h_old, C_old, LSTM)
z = [h_old;x]; % concat inputs

% intermediates
forget = sig((LSTM.forgetWeight * z) + (LSTM.forgetBias));
input = sig((LSTM.inputWeight * z) + (LSTM.inputBias));
C_bar = tanh((LSTM.cellUpdateWeight * z) + (LSTM.cellUpdateBias));

% new parameters
C = (forget .* C_old) + (input .* C_bar);
output = sig((LSTM.outputWeight * z) + LSTM.outputBias);
hidden = (output .* tanh(C));

%output prediction
predict = (LSTM.predictWeight * hidden) + LSTM.predictBias;

end


%% Define function for BPTT

function [dhidden_old, dC_old, LSTM] = backward(target, dhidden_next,...
    dC_next, C_old, z, forget, input, C_bar, C, output, ...
    hidden, predict, LSTM)

% Calculate MSE
dpredict = 0.5*(predict - target).^2;

% Backprop error through the network, starting at output
% Prediction -> output
LSTM.predictWeightD = LSTM.predictWeightD + (dpredict * hidden');
LSTM.predictBiasD = LSTM.predictBiasD + dpredict;

% hidden + output -> output weight
dhidden = (LSTM.predictWeight' * dpredict);
dhidden = dhidden + dhidden_next;
doutput = dhidden .* tanh(C);
doutput = dsig(output) .* doutput;
LSTM.outputWeightD = LSTM.outputWeightD + (doutput * z');
LSTM.outputBiasD = LSTM.outputBiasD + doutput;

%  output + C_bar -> cell update weight
dC = dC_next;
dC = dC + dhidden .* output .* dtanh(tanh(C));
dC_bar = dC .* input;
dC_bar = dtanh(dC_bar) .* dC_bar;
LSTM.cellUpdateWeightD = (dC_bar * z');
LSTM.cellUpdateBiasD = LSTM.cellUpdateBiasD + dC_bar;

% input + cell state -> input weight
dinput = dC .* C_bar;
dinput = dsig(input) .* dinput;
LSTM.inputWeightD = LSTM.inputWeightD + (dinput * z');
LSTM.inputBiasD = LSTM.inputBiasD + dinput;

% old cell state + forget -> forget weight 
dforget = dC .* C_old;
dforget = dsig(forget) .* dforget;
LSTM.forgetWeightD = LSTM.forgetWeightD + (dforget * z');
LSTM.forgetBiasD = LSTM.forgetBiasD + dforget;

% find total error to backpropogate in time
dz = (LSTM.forgetWeight' * dforget) + (LSTM.inputWeight' *dinput) + ...
    (LSTM.cellUpdateWeight' * dC_bar) + (LSTM.outputWeight' * doutput);
dhidden_old = dz(1:LSTM.hiddenSize, :);
dC_old = forget .* dC;

end

%% Define regime for data/error propogation

function [loss, predictStore, hiddenStore, CStore, LSTM] = completePass(X, Y, hidden_old, C_old, LSTM)
% Define temp storage vars
XStore = zeros(length(X(:,1)), length(X(1,:)));
ZStore = zeros(length(X(:,1)) + LSTM.hiddenSize, length(X(1,:)));
forgetStore = zeros(LSTM.hiddenSize, length(X(1,:)));
inputStore = zeros(LSTM.hiddenSize, length(X(1,:)));
C_barStore = zeros(LSTM.hiddenSize, length(X(1,:)));
CStore = zeros(LSTM.hiddenSize, length(X(1,:)));
outputStore = zeros(LSTM.hiddenSize, length(X(1,:)));
hiddenStore = zeros(LSTM.hiddenSize, length(X(1,:)));
predictStore = zeros(2, length(X(1,:)));

% prior values - likely to be 0
%XStore(1) = zeros(length(X(:,1)),1);
%ZStore(1) = zeros(length(X(:,1)) + LSTM.hiddenSize,1);
%forgetStore(1) = zeros(LSTM.hiddenSize,1);
%inputStore(1) = zeros(LSTM.hiddenSize,1);
%C_barStore(1) = zeros(LSTM.hiddenSize,1);
%CStore(1) = zeros(LSTM.hiddenSize,1);
%outputStore(1) = zeros(LSTM.hiddenSize,1);

%hiddenStore(1) = hidden_old;
%CStore(1) = C_old;

% init loss
loss = 0;


%forward prop input data and store intermediate states for backprop
for t = 2:length(X(1,:))
    XStore(:,t) = X(:,t);
    [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
    forward(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTM);
    
    loss = loss + 0.5*(predictStore(:,t) - Y(:,t)).^2; 
    %disp(t)
end
% zero gradients before backprop 
hiddenLayers = LSTM.hiddenSize;
concatLayers = 98 + hiddenLayers;
LSTM.forgetWeightD = zeros(hiddenLayers, concatLayers);
LSTM.forgetBiasD = zeros(hiddenLayers,1);

LSTM.inputWeightD = zeros(hiddenLayers, concatLayers);
LSTM.inputBiasD = zeros(hiddenLayers,1);

LSTM.cellUpdateWeightD = zeros(hiddenLayers, concatLayers);
LSTM.cellUpdateBiasD = zeros(hiddenLayers,1);

LSTM.outputWeightD = zeros(hiddenLayers, concatLayers);
LSTM.outputBiasD = zeros(hiddenLayers,1);

LSTM.predictWeightD = zeros(2, hiddenLayers);
LSTM.predictBiasD = zeros(2,1);

% zero error to be propogated
%disp('zeroed init!')
dhiddenNext = zeros(hiddenLayers,  1);
dCNext = zeros(hiddenLayers, 1);

% backwards pass - BPTT
for t = length(X(1,:)):-1:2
    [dhiddenNext, dCNext, LSTM] = backward(Y(:,t), dhiddenNext, dCNext, CStore(:,t-1), ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), ...
        CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t), LSTM);
    %disp('backprop!')
    %TO DO - clip gradients?
end
% Clip gradients to prevent exploding gradients 
clipUpperThreshold = 5;
clipLowerThreshold = -5;

LSTM.forgetWeightD = min(LSTM.forgetWeightD, clipUpperThreshold);
LSTM.forgetWeightD = max(LSTM.forgetWeightD, clipLowerThreshold);

LSTM.inputWeightD = min(LSTM.inputWeightD, clipUpperThreshold);
LSTM.inputWeightD = max(LSTM.inputWeightD, clipLowerThreshold);

LSTM.cellUpdateWeightD = min(LSTM.cellUpdateWeightD, clipUpperThreshold);
LSTM.cellUpdateWeightD = max(LSTM.cellUpdateWeightD, clipLowerThreshold);

LSTM.outputWeightD = min(LSTM.outputWeightD, clipUpperThreshold);
LSTM.outputWeightD = max(LSTM.outputWeightD, clipLowerThreshold);

LSTM.predictWeightD = min(LSTM.predictWeightD, clipUpperThreshold);
LSTM.predictWeightD = max(LSTM.predictWeightD, clipLowerThreshold);

end

%% Update Parameters

function LSTM = step(LSTM, eta)


LSTM.forgetWeightM = LSTM.forgetWeightM +  (LSTM.forgetWeightD .* LSTM.forgetWeightD);      %Update momentum for next update
LSTM.forgetWeight = (LSTM.forgetWeight - (eta .* LSTM.forgetWeightD) ./ ((LSTM.forgetWeightM + eps(1)).^0.5)); %eps(1) = epsilon 

LSTM.forgetBiasM = LSTM.forgetBiasM + (LSTM.forgetBiasD .* LSTM.forgetBiasD);     
LSTM.forgetBias = (LSTM.forgetBias - (eta .* LSTM.forgetBiasD) ./ ((LSTM.forgetBiasM + eps(1)).^0.5)); 

LSTM.inputWeightM = LSTM.inputWeightM + (LSTM.inputWeightD .* LSTM.inputWeightD);     
LSTM.inputWeight = (LSTM.inputWeight - (eta .* LSTM.inputWeightD) ./ ((LSTM.inputWeightM + eps(1)).^0.5)); 

LSTM.inputBiasM = LSTM.inputBiasM + (LSTM.inputBiasD .* LSTM.inputBiasD);     
LSTM.inputBias = (LSTM.inputBias - (eta .* LSTM.inputBiasD) ./ ((LSTM.inputBiasM + eps(1)).^0.5)); 

LSTM.cellUpdateWeightM = LSTM.cellUpdateWeightM + (LSTM.cellUpdateWeightD .* LSTM.cellUpdateWeightD);     
LSTM.cellUpdateWeight = (LSTM.cellUpdateWeight - (eta .* LSTM.cellUpdateWeightD) ./ ((LSTM.cellUpdateWeightM + eps(1)).^0.5)); 

LSTM.cellUpdateBiasM = LSTM.cellUpdateBiasM + (LSTM.cellUpdateBiasD .* LSTM.cellUpdateBiasD);     
LSTM.cellupdateBias = (LSTM.cellUpdateBias - (eta .* LSTM.cellUpdateBiasD) ./ ((LSTM.cellUpdateBiasM + eps(1)).^0.5)); 

LSTM.outputWeightM = LSTM.outputWeightM + (LSTM.outputWeightD .* LSTM.outputWeightD);     
LSTM.outputWeight = (LSTM.outputWeight - (eta .* LSTM.outputWeightD) ./ ((LSTM.outputWeightM + eps(1)).^0.5)); 

LSTM.outputBiasM = LSTM.outputBiasM + (LSTM.outputBiasD .* LSTM.outputBiasD);     
LSTM.outputBias = (LSTM.outputBias - (eta .* LSTM.outputBiasD) ./ ((LSTM.outputBiasM + eps(1)).^0.5)); 

LSTM.predictWeightM = LSTM.predictWeightM + (LSTM.predictWeightD .* LSTM.predictWeightD);     
LSTM.predictWeight = (LSTM.predictWeight - (eta .* LSTM.predictWeightD) ./ ((LSTM.predictWeightM + eps(1)).^0.5)); 

LSTM.predictBiasM = LSTM.predictBiasM + (LSTM.predictBiasD .* LSTM.predictBiasD);     
LSTM.predictBias = (LSTM.predictBias - (eta .* LSTM.predictBiasD) ./ ((LSTM.predictBiasM + eps(1)).^0.5)); 

end



