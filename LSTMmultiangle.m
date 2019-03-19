%% Load data
clear all; close all;
load('monkeydata_training.mat');

% set random number generator
rng(2013);

% shuffle the data
ix = randperm(length(trial));

XTrain = {};
for angle = 1:8
    for i = 1:80
       XTrain{angle,i} = trial(i,angle).spikes; 
       YTrain{angle,i} = trial(i,angle).handPos(1:2,:);
    end
end
for angle = 1:8
    for i = 1:80
       XTest{angle,i} = trial(i,angle).spikes; 
       YTest{angle,i} = trial(i,angle).handPos(1:2,:);
    end
end

timeStep = 20;
%%
% split into train and test sets
%{
trainTestSplit = 0.8;
numTrain = trainTestSplit * length(trial);
numTest = length(trial)-trainTestSplit * length(trial);

trainingData = trial(ix(1:numTrain),:);
testData = trial(ix(numTrain+1:end),:);




  
timeStep = 5; % in samples
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


XTrainCount = cell(size(intermediate, 3),1);
YTrainCount = cell(size(intermediate, 3),1);
%}
for angle = 1:8
    for trial = 1:length(XTrain(angle,:))
       count = [];
       countTrack = 1;
       idxTrack = 1;
       temp = XTrain{angle,trial};

       while(idxTrack+timeStep < length(temp(1,:)))
           count(:,countTrack) = (1/timeStep)* sum(temp(:,idxTrack:idxTrack+(timeStep-1)),2);
           countTrack = countTrack +1;
           idxTrack = idxTrack+timeStep;
       end
       %count(:,countTrack+1) = [count [sum(temp(:,(idxTrack+1-timeStep):end) zeros(98, 5-(length(temp(1,:)) - idxTrack)),2)]];
       if((idxTrack - length(temp(1,:)))~=0)
           count = [count zeros(98,1)];
       end
       XTrainCount{angle,trial} = count;

    end
end
for angle = 1:8
    
    for trial = 1:length(YTrain(angle,:))
       count = [];
       countTrack = 1;
       idxTrack = 1;
       temp = YTrain{angle,trial};
       %{
       while(idxTrack+timeStep < length(temp(1,:)))
           posX = temp(1,idxTrack:idxTrack+(timeStep-1));
           posY = temp(2,idxTrack:idxTrack+(timeStep-1));
           pos = [posX'; posY'];
           count(:,countTrack) = pos;
           countTrack = countTrack +1;
           idxTrack = idxTrack+timeStep;
       end
       %count(:,countTrack+1) = [count [sum(temp(:,(idxTrack+1-timeStep):end) zeros(98, 5-(length(temp(1,:)) - idxTrack)),2)]];
       if((idxTrack - length(temp(1,:)))~=0)
           posX = temp(1,end)*ones(1,);
           posY = temp(2,end)*ones(1,5);
           count = [count [posX';posY']];
       end
       %}
       while(idxTrack+timeStep < length(temp(1,:)))
          posX = temp(1,idxTrack+(timeStep-1));
          posY = temp(2,idxTrack+(timeStep-1));
          pos = [posX'; posY'];
          count(:,countTrack) = pos;
          countTrack = countTrack +1;
          idxTrack = idxTrack+timeStep;
       end
       %count(:,countTrack+1) = [count [sum(temp(:,(idxTrack+1-timeStep):end) zeros(98, 5-(length(temp(1,:)) - idxTrack)),2)]];
       if((idxTrack - length(temp(1,:)))~=0)
           posX = temp(1,end);
           posY = temp(2,end);
           count = [count [posX';posY']];
       YTrainCount{angle,trial} = count;
       end
    end
end
%{
for trial = 1:length(XTrain)
    neuronStore = [];
    temp = XTrain{trial};
    div = mod(length(temp(1,:)),timeStep);
    count = [];
    
    if(div ~= 0)
       temp = [temp zeros(98, timeStep-div)]; % zero padding to make div by timestep
    end
    
    for neuron = 1:98
       for i = 1:length(temp)/5
          Count(i) = (1/timeStep).* sum(temp(neuron, ((i-1)*5 +1 : (i-1)*5 +5))); 
       end
       neuronStore(neuron,:) = Count;
    end
    
    %t = 1:timeStep:length(temp(1,:));
    %for neuron = 1:98
    %    for i = 1:length(t)-1
    %        Count(i) = (1/timeStep).* sum(sum(temp(neuron,t(i):t(i+1)-1)));
    %    end
  %
 %   end
    XTrainCount{trial} = neuronStore;
    
end
%%

for trial = 1:length(YTrain)
    temp = YTrain{trial};
    posStore = [];
    div = mod(length(temp(1,:)),timeStep);
    if(div ~= 0)
       finalX = temp(1,end);
       finalY = temp(2,end);
       appendVector = [finalX*ones(1,timeStep-div) ;finalY*ones(1,timeStep-div)]; % Pad with final value
       temp = [temp appendVector]; 
       %disp('in loop')
    end
    
    t = 1:timeStep:length(temp(1,:));
    for i = 1:length(t)-1
       idx = t(i):t(i+1)-1;
       posStoreX = [temp(1,idx)];
       posStoreY = [temp(2,idx)];
       posStore(:,i) = [posStoreX'; posStoreY'];
    end
    %posStoreX = temp(1,(idx(end)+1:end)); % Pad the ends of sequences
    %posStoreY = temp(2,(idx(end)+1:end)); % that aren't divisible by 5
    %for j = 1:4
    %   if(length(posStoreX) == j)
    %       posStoreX(j:5) = posStoreX(j);
    %       posStoreY(j:5) = posStoreY(j);
    %       XTrainCount{trial} = [XTrainCount{trial} zeros(98, 5)]; %Zero pad
    %   end
    %end
    %posStore(:,i+1) = [posStoreX'; posStoreY'];
    
    YTrainCount{trial} = posStore;
   
end
%}
%%
%{
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

%}
%% Define parameters
for angle = 1:8
% global
eta = 0.005;
inputSize = 98;
noOutputs = 2;
% LSTM parameters
hiddenLayers = 100;
concatLayers = inputSize + hiddenLayers;

forgetWeight = 0.001*randn(hiddenLayers, concatLayers)+0.5;
forgetBias = 0.001*randn(hiddenLayers,1)+0.5;

inputWeight = 0.001*randn(hiddenLayers, concatLayers)+0.5;
inputBias = 0.001*randn(hiddenLayers, 1)+0.5;

cellUpdateWeight = 0.001*randn(hiddenLayers, concatLayers);
cellUpdateBias = 0.001*randn(hiddenLayers,1);

outputWeight = 0.001*randn(hiddenLayers, concatLayers)+0.5;
outputBias = 0.001*randn(hiddenLayers,1);

predictWeight = 0.001*randn(noOutputs, hiddenLayers)+0.5;
predictBias = 0.001*randn(noOutputs, 1);

LSTM = struct;

LSTM.noOutputs = noOutputs;

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
LSTM.predictWeightD = zeros(LSTM.noOutputs, hiddenLayers);
LSTM.predictWeightM = zeros(LSTM.noOutputs, hiddenLayers);

LSTM.predictBias =  predictBias;
LSTM.predictBiasD = zeros(LSTM.noOutputs,1);
LSTM.predictBiasM = zeros(LSTM.noOutputs,1);

%% Training Loop - batch size 1 (SGD, Adagrad)
maxEpochs = 1000;

totalLoss = zeros(1,maxEpochs);
% scale positional data to reduce exploding gradients


for epoch = 1:maxEpochs
    for trial = 1:length(XTrainCount(angle,:))
        % Set t-1 LSTM memory states
        hidden_init = zeros(LSTM.hiddenSize,1);
        C_init = zeros(LSTM.hiddenSize,1);
        % Extract info from cells
        XTrial = XTrainCount{angle,trial};
        YTrial = YTrainCount{angle,trial}; % scaling to reduce explosions

        [loss,predictStore, hidden, C, LSTM] = completePass(XTrial, YTrial, hidden_init, C_init, LSTM);
        totalLoss(:,epoch) = 0.1 * (sum(loss).^0.5);
        
        LSTM = step(LSTM, eta);
        disp("Epoch:")
        disp(epoch)
        disp("Trial:")
        disp(trial)
    end
angleLoss(angle,:) = totalLoss;
LSTMStore{angle} = LSTM;
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

% Calculate Error
dpredict = -0.5*(predict - target).^2;

% Clip MSE
dC_barUpper = 10000;
dC_barLower = -10000;
%dpredict = min(50, dpredict);

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
% clip dC_bar
dC_bar = min(dC_bar, dC_barUpper);
dC_bar = max(dC_bar, dC_barLower);
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
predictStore = zeros(LSTM.noOutputs, length(X(1,:)));

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
for t = 1:length(X(1,:))
    if (t == 1)
        XStore(:,t) = X(:,t);
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
        forward(X(:,t), zeros(LSTM.hiddenSize,1), zeros(LSTM.hiddenSize,1), LSTM);    
    
    else
        XStore(:,t) = X(:,t);
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
        forward(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTM);
    end
    loss = loss + 0.1*(predictStore(:,t) - Y(:,t)).^2; 
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

LSTM.predictWeightD = zeros(LSTM.noOutputs, hiddenLayers);
LSTM.predictBiasD = zeros(LSTM.noOutputs,1);

% zero error to be propogated
%disp('zeroed init!')
dhiddenNext = zeros(hiddenLayers,  1);
dCNext = zeros(hiddenLayers, 1);

% backwards pass - BPTT
for t = length(X(1,:)):-1:1
    if(t == 1)
    [dhiddenNext, dCNext, LSTM] = backward(Y(:,t), dhiddenNext, dCNext, zeros(LSTM.hiddenSize,1), ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), ...
        CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t), LSTM);
    else
    [dhiddenNext, dCNext, LSTM] = backward(Y(:,t), dhiddenNext, dCNext, CStore(:,t-1), ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), ...
        CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t), LSTM);
    end
    %disp('backprop!')
    %TO DO - clip gradients?
end
% Clip gradients to prevent exploding gradients 
clipUpperThreshold = 10;
clipLowerThreshold = -10;

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
LSTM.forgetWeight = (LSTM.forgetWeight - (eta .* LSTM.forgetWeightD) ./ ((LSTM.forgetWeightM + 1e-8).^0.5)); %eps(1) = epsilon

LSTM.forgetBiasM = LSTM.forgetBiasM + (LSTM.forgetBiasD .* LSTM.forgetBiasD);     
LSTM.forgetBias = (LSTM.forgetBias - (eta .* LSTM.forgetBiasD) ./ ((LSTM.forgetBiasM + 1e-8).^0.5)); 

LSTM.inputWeightM = LSTM.inputWeightM + (LSTM.inputWeightD .* LSTM.inputWeightD);     
LSTM.inputWeight = (LSTM.inputWeight - (eta .* LSTM.inputWeightD) ./ ((LSTM.inputWeightM + 1e-8).^0.5)); 

LSTM.inputBiasM = LSTM.inputBiasM + (LSTM.inputBiasD .* LSTM.inputBiasD);     
LSTM.inputBias = (LSTM.inputBias - (eta .* LSTM.inputBiasD) ./ ((LSTM.inputBiasM + 1e-8).^0.5)); 

LSTM.cellUpdateWeightM = LSTM.cellUpdateWeightM + (LSTM.cellUpdateWeightD .* LSTM.cellUpdateWeightD);     
LSTM.cellUpdateWeight = (LSTM.cellUpdateWeight - (eta .* LSTM.cellUpdateWeightD) ./ ((LSTM.cellUpdateWeightM + 1e-8).^0.5)); 

LSTM.cellUpdateBiasM = LSTM.cellUpdateBiasM + (LSTM.cellUpdateBiasD .* LSTM.cellUpdateBiasD);     
LSTM.cellupdateBias = (LSTM.cellUpdateBias - (eta .* LSTM.cellUpdateBiasD) ./ ((LSTM.cellUpdateBiasM + 1e-8).^0.5)); 

LSTM.outputWeightM = LSTM.outputWeightM + (LSTM.outputWeightD .* LSTM.outputWeightD);     
LSTM.outputWeight = (LSTM.outputWeight - (eta .* LSTM.outputWeightD) ./ ((LSTM.outputWeightM + 1e-8).^0.5)); 

LSTM.outputBiasM = LSTM.outputBiasM + (LSTM.outputBiasD .* LSTM.outputBiasD);     
LSTM.outputBias = (LSTM.outputBias - (eta .* LSTM.outputBiasD) ./ ((LSTM.outputBiasM + 1e-8).^0.5)); 

LSTM.predictWeightM = LSTM.predictWeightM + (LSTM.predictWeightD .* LSTM.predictWeightD);     
LSTM.predictWeight = (LSTM.predictWeight - (eta .* LSTM.predictWeightD) ./ ((LSTM.predictWeightM + 1e-8).^0.5)); 

LSTM.predictBiasM = LSTM.predictBiasM + (LSTM.predictBiasD .* LSTM.predictBiasD);     
LSTM.predictBias = (LSTM.predictBias - (eta .* LSTM.predictBiasD) ./ ((LSTM.predictBiasM + 1e-8).^0.5)); 

%{
LSTM.forgetWeight = LSTM.forgetWeight + eta*LSTM.forgetWeightD;
LSTM.forgetBias = LSTM.forgetBias + eta*LSTM.forgetBiasD;

LSTM.outputWeight = LSTM.outputWeight + eta*LSTM.outputWeightD;
LSTM.outputBias = LSTM.outputBias + eta*LSTM.outputBiasD;

LSTM.cellUpdateWeight = LSTM.cellUpdateWeight + eta*LSTM.cellUpdateWeightD;
LSTM.cellUpdateBias = LSTM.cellUpdateBias + eta*LSTM.cellUpdateBiasD;

LSTM.inputWeight = LSTM.inputWeight + eta*LSTM.inputWeightD;
LSTM.inputBias = LSTM.inputBias + eta*LSTM.inputBiasD;

LSTM.predictWeight = LSTM.predictWeight + eta*LSTM.predictWeightD;
LSTM.predictBias = LSTM.predictBias + eta*LSTM.predictBiasD;
end
%}
end
%% SEPBLOCK _ NEED TO IMPLEMENT


