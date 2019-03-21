function [x, y] = positionEstimator(test_data, modelParameters)
%% Normalise test data

meanTrajX = modelParameters.meanTrajX;
meanTrajY = modelParameters.meanTrajY;

LSTMStore = modelParameters.LSTMStore;

XRaw = test_data.spikes;
predict = {};
timeStep = 20;
%% Bin the spike data


X = [];
countTrack = 1;
idxTrack = 1;


while(idxTrack+timeStep < length(XRaw(1,:)))
    X(:,countTrack) = (1/timeStep)* sum(XRaw(:,idxTrack:idxTrack+(timeStep-1)),2);
    countTrack = countTrack +1;
    idxTrack = idxTrack+timeStep;
end
%count(:,countTrack+1) = [count [sum(temp(:,(idxTrack+1-timeStep):end) zeros(98, 5-(length(temp(1,:)) - idxTrack)),2)]];
if((idxTrack - length(XRaw(1,:)))~=0) % Zero pad if length not div by timeStep
    X = [X zeros(98,1)];      % i.e. leftover data
end

%% Determine most likely angle

classifer = modelParameters.classifer;
hidden_init = zeros(classifer.hiddenSize,1);
C_init = zeros(classifer.hiddenSize,1);

for t = 1: min(length(X(1,:)), 20) % classifer trained up to 20 time steps due to BPTT limitations
    if (t == 1)
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
            forwardClass(X(:,t), zeros(classifer.hiddenSize,1), zeros(classifer.hiddenSize,1), classifer);

    else
        XStore(:,t) = X(:,t);
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
            forwardClass(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), classifer);
    end
end

%{
for t = 1: length(X(1,:)) % classifer trained up to 20 time steps due to BPTT limitations
    if (t == 1)
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
            forwardClass(X(:,t), zeros(classifer.hiddenSize,1), zeros(classifer.hiddenSize,1), classifer);

    else
        XStore(:,t) = X(:,t);
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
            forwardClass(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), classifer);
    end
end
%}
[~,angle] = max(sum(predictStore(:,8:end),2));

%% Get trajectory


for coord = 1:2
    LSTMtemp = LSTMStore{angle,coord};
    LSTM = LSTMtemp;
    hidden_init = zeros(LSTM.hiddenSize,1);
    ZStore = [];
    forgetStore = [];
    inputStore = [];
    C_barStore = [];
    CStore = [];
    outputStore = [];
    hiddenStore = [];
    predictStore = [];
    
    C_init = zeros(LSTM.hiddenSize,1);
    for t = 1:length(X(1,:))
    
        if (t == 1)
            [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
                forward(X(:,t), zeros(LSTM.hiddenSize,1), zeros(LSTM.hiddenSize,1), LSTM);
            
        else
            XStore(:,t) = X(:,t);
            [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
                forward(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTM);
        end
    end
    predict{coord} = predictStore;
end

%% Get predictions for each LSTM
%{
for angle = 1:8
    for coord = 1:2
        LSTMtemp = LSTMStore{angle,coord};
        LSTM = LSTMtemp;
        for t = 1:length(X(1,:))
            if (t == 1)
                [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
                    forward(X(:,t), zeros(LSTM.hiddenSize,1), zeros(LSTM.hiddenSize,1), LSTM);
                
            else
                XStore(:,t) = X(:,t);
                [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
                    forward(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTM);
            end
        end
        predict{angle, coord} = predictStore;
    end
end
%}
%% Find the MSE between each LSTM prediction and the mean traj for the that angle
%{
% mostly stationary in the first 320ms so only use 240-320ms 
% i.e samples 11-16
MSE = zeros(1,8);
for angle = 1:8
    predictX = predict{angle,1};
    predictY = predict{angle,2};
    
    diffX = predictX(end) - predict
    
    sumLength = min(length(predictX), length(meanTrajX));
    
    
    MSEX = (predictX(16) - meanTrajX(angle,16)).^2;
    MSEY = (predictY(16) - meanTrajY(angle,16)).^2;
    MSE(angle) = sum(MSEX) + sum(MSEY);
end

[~, predictedAngle] = min(MSE);
%}
%% Take last predicted value from LSTM with most likely angle and transform

%predictX = predict{predictedAngle, 1};
%predictY = predict{predictedAngle, 2};
predictX = predict{1};
predictY = predict{2};
x = (modelParameters.maxposX - modelParameters.minposX).* predictX(end) + modelParameters.meanposX;
y = (modelParameters.maxposY - modelParameters.minposY).* predictY(end) + modelParameters.meanposY;



end

%% Other functions - sigmoid, softmax and forward pass 

function output = sig(input)
output = 1./(1 + exp(-input));
end

% softmax
function output = softmax(input)
output = exp(input) ./ (sum(exp(input)));
end

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
predict = tanh(predict);
end

%% CLASSIFER FUNCTIONS - forward pass
function [z,forget,input,C_bar, C,output,hidden,predict] = ...
    forwardClass(x, h_old, C_old, LSTM)
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
predict = softmax(predict);
end

