function [modelParameters] = positionEstimatorTraining(training_data)

% Determine if data should be loaded or produced:
if exist('modelParametersAlt.mat', 'file') == 2
    load('modelParametersAlt.mat');
    modelParameters = modelParametersAlt;
else
    % define number of samples in each bin
    timeStep = 20;
    modelParameters = struct;
    % set random number generator
    rng(2013);
    
    
    
    XTrain = {};
    for angle = 1:8
        for i = 1:length(training_data(:,1))
            XTrain{angle,i} = training_data(i,angle).spikes;
            YTrain{angle,i} = training_data(i,angle).handPos(1:2,:);
        end
    end
    
    
    
    %% Norm Ydata to [-1,1] using mean normalisation
    
    % X data is already [-1,1] so does not need to be normalised
    
    % init variables
    meanposX = 0;
    meanposY = 0;
    stdposX = 0;
    stdposY = 0;
    maxposX = 0;
    maxposY = 0;
    minposX = 0;
    minposY = 0;
    
    % iter through angles and trials - build cummulative mean & std, find
    % min/max
    for angle = 1:8
        for i = 1:length(training_data(:,1))
            temp = YTrain{angle,i};
            meanposX = meanposX + mean(temp(1,:));
            meanposY = meanposY + mean(temp(2,:));
            
            stdposX = stdposX + std(temp(1,:));
            stdposY = stdposY + std(temp(2,:));
            
            tempminX = min(temp(1,:));
            tempmaxX = max(temp(1,:));
            tempminY = min(temp(2,:));
            tempmaxY = max(temp(2,:));
            
            if(tempminX < minposX)
                minposX = tempminX;
            end
            
            if(tempmaxX > maxposX)
                maxposX = tempmaxX;
            end
            
            if(tempminY < minposY)
                minposY = tempminY;
            end
            
            if(tempmaxY > maxposY)
                maxposY = tempmaxY;
            end
            
        end
    end
    
    meanposX = meanposX/(8*length(training_data(:,1)));
    meanposY = meanposY/(8*length(training_data(:,1)));
    
    stdposX = stdposX/8*length(training_data(:,1));
    stdposY = stdposY/8*length(training_data(:,1));
    
    modelParameters.meanposX = meanposX;
    modelParameters.meanposY = meanposY;
    modelParameters.stdposX = stdposX;
    modelParameters.stdposY = stdposY;
    modelParameters.maxposX = maxposX;
    modelParameters.minposx = minposX;
    modelParameters.maxposY = maxposY;
    modelParameters.minposY = minposY;
    %% Apply mean norm to Ydata
    
    for angle = 1:8
        for i = 1:length(training_data(:,1))
            temp = YTrain{angle,i};
            temp(1,:) = (1/(maxposX - minposX))*(temp(1,:) - meanposX);
            temp(2,:) = (1/(maxposY - minposY))*(temp(2,:) - meanposY);
            YTrain{angle,i} = temp;
        end
    end
    
    
    %% XTrain Binning
    
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
            if((idxTrack - length(temp(1,:)))~=0) % Zero pad if length not div by timeStep
                count = [count zeros(98,1)];      % i.e. leftover data
            end
            XTrainCount{angle,trial} = count;
            
        end
    end
    
    %% YTrain Binning
    
    for angle = 1:8
        
        for trial = 1:length(YTrain(angle,:))
            count = [];
            countTrack = 1;
            idxTrack = 1;
            temp = YTrain{angle,trial};
            
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
                count = [count [posX';posY']]; % Pad with final value
                YTrainCount{angle,trial} = count;
            end
        end
    end
    
    %% XTest Binning
    %{
    for angle = 1:8
        for trial = 1:length(XTest(angle,:))
           count = [];
           countTrack = 1;
           idxTrack = 1;
           temp = XTest{angle,trial};

           while(idxTrack+timeStep < length(temp(1,:)))
               count(:,countTrack) = (1/timeStep)* sum(temp(:,idxTrack:idxTrack+(timeStep-1)),2);
               countTrack = countTrack +1;
               idxTrack = idxTrack+timeStep;
           end
           %count(:,countTrack+1) = [count [sum(temp(:,(idxTrack+1-timeStep):end) zeros(98, 5-(length(temp(1,:)) - idxTrack)),2)]];
           if((idxTrack - length(temp(1,:)))~=0)
               count = [count zeros(98,1)];
           end
           XTestCount{angle,trial} = count;

        end
    end




    %% YTest Binning

    for angle = 1:8

        for trial = 1:length(YTest(angle,:))
           count = [];
           countTrack = 1;
           idxTrack = 1;
           temp = YTest{angle,trial};

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
           YTestCount{angle,trial} = count;
           end
        end
    end

    %}
    %% Find mean norm trajectories for determining which LSTM to use in the testing phase
    %{
% only use first 300ms as this is the fewest amount of data points
meanTrajX = zeros(8,29);
meanTrajY = zeros(8,29);

for angle = 1:8
    for i = 1:length(training_data(:,1))
        temp = YTrainCount{angle,i};
        meanTrajX(angle,:) = meanTrajX(angle,:) + (temp(1,:));
        meanTrajY(angle,:) = meanTrajY(angle,:) + (temp(2,:));
    end
    meanTrajX(angle,:) = meanTrajX(angle,:)/100;
    meanTrajY(angle,:) = meanTrajY(angle,:)/100;
end
modelParameters.meanTrajX = meanTrajX;
modelParameters.meanTrajY = meanTrajY;
    %}
    %% Define parameters - loop through angle + [x,y] to train 16 LSTMs
    for angle = 1:8
        for coord = 1:2
            % global
            eta = 0.1;%eta = 0.05;
            inputSize = 98;
            noOutputs = 1;
            % LSTM parameters
            hiddenLayers = 10;
            concatLayers = inputSize + hiddenLayers;
            weightScale = 0.0001;
            
            forgetWeight = weightScale*randn(hiddenLayers, concatLayers)+0.5;
            forgetBias = 0.1*weightScale*randn(hiddenLayers,1)+0.5;
            
            inputWeight = weightScale*randn(hiddenLayers, concatLayers)+0.5;
            inputBias = 0.1*weightScale*randn(hiddenLayers, 1)+0.5;
            
            cellUpdateWeight = weightScale*randn(hiddenLayers, concatLayers);
            cellUpdateBias = 0.1*weightScale*randn(hiddenLayers,1);
            
            outputWeight = weightScale*randn(hiddenLayers, concatLayers)+0.5;
            outputBias = 0.1*weightScale*randn(hiddenLayers,1);
            
            predictWeight = weightScale*randn(noOutputs, hiddenLayers)+0.5;
            predictBias = 0.1*weightScale*randn(noOutputs, 1);
            
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
            maxEpochs = 5000; % Seems large, but needed for long sequence + BPTT + AdaGra
            
            totalLoss = zeros(1,maxEpochs);
            % scale positional data to reduce exploding gradients
            
            
            for epoch = 1:maxEpochs
                for trial = 1:length(XTrainCount(angle,:))
                    % Set t-1 LSTM memory states
                    hidden_init = zeros(LSTM.hiddenSize,1);
                    C_init = zeros(LSTM.hiddenSize,1);
                    % Extract info from cells
                    XTrial = XTrainCount{angle,trial};
                    YTrial = YTrainCount{angle,trial};
                    YTrial = YTrial(coord, :);% scaling to reduce explosions
                    
                    [loss,predictStore, hidden, C, LSTM] = completePass(XTrial, YTrial, hidden_init, C_init, LSTM);
                    totalLoss(epoch) = totalLoss(epoch) + (sum(loss));
                    
                    LSTM = step(LSTM, eta);
                    disp("Epoch:")
                    disp(epoch)
                    disp("Trial:")
                    disp(trial)
                end
                %coordLoss(coord,:) = totalLoss;
                %angleLoss(angle,:) = totalLoss;
                LSTMStore{angle,coord} = LSTM; % Store LSTM
                
                
                if(mod(epoch, 500) ==0) % Learning Rate Scheduler to refine
                    eta =eta * 0.95;
                end
            end
            
        end
    end
    
    %% Train Classifer
    
    eta = 0.01;%eta = 0.05;
    inputSize = 98;
    noOutputs = 8;
    % LSTM parameters
    hiddenLayers = 50;
    concatLayers = inputSize + hiddenLayers;
    weightScale = 0.01;
    
    forgetWeight = weightScale*randn(hiddenLayers, concatLayers)+0.5;
    forgetBias = 0*weightScale*randn(hiddenLayers,1);
    
    inputWeight = weightScale*randn(hiddenLayers, concatLayers)+0.5;
    inputBias = 0*weightScale*randn(hiddenLayers, 1);
    
    cellUpdateWeight = weightScale*randn(hiddenLayers, concatLayers);
    cellUpdateBias = 0*weightScale*randn(hiddenLayers,1);
    
    outputWeight = weightScale*randn(hiddenLayers, concatLayers)+0.5;
    outputBias = 0*weightScale*randn(hiddenLayers,1);
    
    predictWeight = weightScale*randn(noOutputs, hiddenLayers)+0.5;
    predictBias = 0*weightScale*randn(noOutputs, 1);
    
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
    
    %% Training Loop - batch size 1 (SGD, Adagrad), different learning rate
    maxEpochs = 5000; % Seems large, but needed for long sequence + BPTT + AdaGra
    
    totalLoss = zeros(1,maxEpochs);
    % scale positional data to reduce exploding gradients
    
    
    for epoch = 1:maxEpochs
        for trial = 1:length(XTrainCount(1,:))
            for angle = randperm(8)
                
                
                % Set t-1 LSTM memory states
                hidden_init = zeros(LSTM.hiddenSize,1);
                C_init = zeros(LSTM.hiddenSize,1);
                % Extract info from cells
                XTrial = XTrainCount{angle,trial};
                %YTrial = YTrainCount{angle,trial};
                %YTrial = YTrial(coord, :);% scaling to reduce explosions
                oneHot = zeros(8, length(XTrial));
                oneHot(angle,:) = 1;
                
                
                [loss,predictStore, hidden, C, LSTM] = completePassClass(XTrial, oneHot, hidden_init, C_init, LSTM);
                totalLoss(epoch) = totalLoss(epoch) + (sum(loss));
                
                LSTM = step(LSTM, eta);
                disp("Epoch:")
                disp(epoch)
                disp("Trial:")
                disp(trial)
            end
        end
        %coordLoss(coord,:) = totalLoss;
        %angleLoss(angle,:) = totalLoss;
        
        
        
        if(mod(epoch, 500) ==0) % Learning Rate Scheduler to refine
            eta =eta * 0.95;
        end
    end
    
    %save('LSTMxy2.mat', 'LSTMStore');
    
    %  modelParameters.traj = [meanTrajX; meanTrajY];
    modelParameters.LSTMStore = LSTMStore;
    modelParameters.classifer = LSTM;
    %% Testing
    %{
    X = XTestCount{1,4};
    YTrial = YTestCount{1,4};
    actualX = YTrial(1,:);
    actualY = YTrial(2,:);

    LSTMX = LSTMStore{1,1};
    LSTMY = LSTMStore{1,2};

    XStore = zeros(length(X(:,1)), length(X(1,:)));
    ZStore = zeros(length(X(:,1)) + LSTM.hiddenSize, length(X(1,:)));
    forgetStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    inputStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    C_barStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    CStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    outputStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    hiddenStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    predictStoreX = zeros(LSTM.noOutputs, length(X(1,:)));


    for t = 1:length(X(1,:))
        if (t == 1)
            XStore(:,t) = X(:,t);
            [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStoreX(:,t)] = ...
            forward(X(:,t), zeros(LSTM.hiddenSize,1), zeros(LSTM.hiddenSize,1), LSTMX);

        else
            XStore(:,t) = X(:,t);
            [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStoreX(:,t)] = ...
            forward(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTMX);
        end
        %loss = loss + 0.5*(predictStoreX(:,t) - Y(:,t)).^2;
        %disp(t)
    end

    XStore = zeros(length(X(:,1)), length(X(1,:)));
    ZStore = zeros(length(X(:,1)) + LSTM.hiddenSize, length(X(1,:)));
    forgetStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    inputStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    C_barStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    CStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    outputStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    hiddenStore = zeros(LSTM.hiddenSize, length(X(1,:)));
    predictStoreY = zeros(LSTM.noOutputs, length(X(1,:)));


    for t = 1:length(X(1,:))
        if (t == 1)
            XStore(:,t) = X(:,t);
            [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStoreY(:,t)] = ...
            forward(X(:,t), zeros(LSTM.hiddenSize,1), zeros(LSTM.hiddenSize,1), LSTMY);

        else
            XStore(:,t) = X(:,t);
            [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStoreY(:,t)] = ...
            forward(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTMY);
        end
        %loss = loss + 0.5*(predictStoreY(:,t) - Y(:,t)).^2;
        %disp(t)
    end
    
    figure
    plot(actualX,actualY)
    hold on
    plot(predictStoreX, predictStoreY)
    hold off
    %}
end

end

%% FUNCTION DEFINITIONS %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define activation functions

% sigmoid
function output = sig(input)
output = 1./(1 + exp(-input));
end

% softmax
function output = softmax(input)
output = exp(input) ./ (sum(exp(input)));
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
predict = tanh(predict);
end


%% Define function for BPTT

function [dhidden_old, dC_old, LSTM] = backward(target, dhidden_next,...
    dC_next, C_old, z, forget, input, C_bar, C, output, ...
    hidden, predict, LSTM)

% Calculate Error
error = sum((predict - target));
dpredict = error .* dtanh(LSTM.predictWeight * hidden);


%dpredict = min(50, dpredict);

% Backprop error through the network, starting at output
% Prediction -> output
LSTM.predictWeightD = LSTM.predictWeightD + (dpredict * hidden');
LSTM.predictBiasD = LSTM.predictBiasD + dpredict;

% hidden + output -> output weight
dhidden = (LSTM.predictWeight' * dpredict) + dhidden_next;
doutput = dhidden .* tanh(C);
doutput = dsig(output) .* doutput;
LSTM.outputWeightD = LSTM.outputWeightD + (doutput * z');
LSTM.outputBiasD = LSTM.outputBiasD + doutput;

%  output + C_bar -> cell update weight
dC = dC_next;
dC = dC + dhidden .* output .* dtanh(tanh(C));
dC_bar = dC .* input;
% clip dC_bar - prevent explosion - probably not needed with norm'd data
dC_barUpper = 100000000;
dC_barLower = -100000000;
dC_bar = min(dC_bar, dC_barUpper);
dC_bar = max(dC_bar, dC_barLower);
dC_bar = dtanh(dC_bar) .* dC_bar;
LSTM.cellUpdateWeightD = (dC_bar * z');
LSTM.cellUpdateBiasD = LSTM.cellUpdateBiasD + dC_bar;

% input + cell state -> input weight
dinput = dsig(input) .* (dC .* C_bar);
LSTM.inputWeightD = LSTM.inputWeightD + (dinput * z');
LSTM.inputBiasD = LSTM.inputBiasD + dinput;

% old cell state + forget -> forget weight

dforget = dsig(forget) .* (dC.* C_old);
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

LSTM.predictWeightD = zeros(LSTM.noOutputs, hiddenLayers);
LSTM.predictBiasD = zeros(LSTM.noOutputs,1);

% zero error to be propogated
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


%% Backward pass

function [dhidden_old, dC_old, LSTM] = backwardClass(target, dhidden_next,...
    dC_next, C_old, z, forget, input, C_bar, C, output, ...
    hidden, predict, LSTM)

% Calculate Error
dpredict = predict - target;
%dpredict = dpredict - dpredict(find(target));

%dpredict = min(50, dpredict);

% Backprop error through the network, starting at output
% Prediction -> output
LSTM.predictWeightD = LSTM.predictWeightD + (dpredict * hidden');
LSTM.predictBiasD = LSTM.predictBiasD + dpredict;

% hidden + output -> output weight
dhidden = (LSTM.predictWeight' * dpredict) + dhidden_next;
doutput = dhidden .* tanh(C);
doutput = dsig(output) .* doutput;
LSTM.outputWeightD = LSTM.outputWeightD + (doutput * z');
LSTM.outputBiasD = LSTM.outputBiasD + doutput;

%  output + C_bar -> cell update weight
dC = dC_next;
dC = dC + dhidden .* output .* dtanh(tanh(C));
dC_bar = dC .* input;
% clip dC_bar - prevent explosion - probably not needed with norm'd data
dC_barUpper = 100000000;
dC_barLower = -100000000;
dC_bar = min(dC_bar, dC_barUpper);
dC_bar = max(dC_bar, dC_barLower);
dC_bar = dtanh(dC_bar) .* dC_bar;
LSTM.cellUpdateWeightD = (dC_bar * z');
LSTM.cellUpdateBiasD = LSTM.cellUpdateBiasD + dC_bar;

% input + cell state -> input weight
dinput = dsig(input) .* (dC .* C_bar);
LSTM.inputWeightD = LSTM.inputWeightD + (dinput * z');
LSTM.inputBiasD = LSTM.inputBiasD + dinput;

% old cell state + forget -> forget weight

dforget = dsig(forget) .* (dC.* C_old);
LSTM.forgetWeightD = LSTM.forgetWeightD + (dforget * z');
LSTM.forgetBiasD = LSTM.forgetBiasD + dforget;

% find total error to backpropogate in time
dz = (LSTM.forgetWeight' * dforget) + (LSTM.inputWeight' *dinput) + ...
    (LSTM.cellUpdateWeight' * dC_bar) + (LSTM.outputWeight' * doutput);
dhidden_old = dz(1:LSTM.hiddenSize, :);
dC_old = forget .* dC;

end
%% Complete Pass
function [loss, predictStore, hiddenStore, CStore, LSTM] = completePassClass(X, Y, hidden_old, C_old, LSTM)
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
            forwardClass(X(:,t), zeros(LSTM.hiddenSize,1), zeros(LSTM.hiddenSize,1), LSTM);
        
    else
        XStore(:,t) = X(:,t);
        [ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t)] = ...
            forwardClass(X(:,t), hiddenStore(:,t-1), CStore(:,t-1), LSTM);
    end
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

LSTM.predictWeightD = zeros(LSTM.noOutputs, hiddenLayers);
LSTM.predictBiasD = zeros(LSTM.noOutputs,1);

% zero error to be propogated
dhiddenNext = zeros(hiddenLayers,  1);
dCNext = zeros(hiddenLayers, 1);

% backwards pass - BPTT
for t = length(X(1,:)):-1:1
    if(t == 1)
        [dhiddenNext, dCNext, LSTM] = backwardClass(Y(:,t), dhiddenNext, dCNext, zeros(LSTM.hiddenSize,1), ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), ...
            CStore(:,t), outputStore(:,t), hiddenStore(:,t), predictStore(:,t), LSTM);
    else
        [dhiddenNext, dCNext, LSTM] = backwardClass(Y(:,t), dhiddenNext, dCNext, CStore(:,t-1), ZStore(:,t), forgetStore(:,t), inputStore(:,t), C_barStore(:,t), ...
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

% Adagrad algorithm for SGD - see http://www.jmlr.org/papers/v12/duchi11a.html
% ADAM optimiser would be preferable, but harder to implement in MATLAB

LSTM.forgetWeightM = LSTM.forgetWeightM +  (LSTM.forgetWeightD .* LSTM.forgetWeightD);      %Update momentum for next update
LSTM.forgetWeight = (LSTM.forgetWeight - (eta .* LSTM.forgetWeightD) ./ ((LSTM.forgetWeightM + 1e-8).^0.5)); %1e-8 is to prevent

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

LSTM.predictWeightM = LSTM.predictWeightM +(LSTM.predictWeightD .* LSTM.predictWeightD);
LSTM.predictWeight = (LSTM.predictWeight - (eta .* LSTM.predictWeightD) ./ ((LSTM.predictWeightM + 1e-8).^0.5));

LSTM.predictBiasM = LSTM.predictBiasM + (LSTM.predictBiasD .* LSTM.predictBiasD);
LSTM.predictBias = (LSTM.predictBias - (eta .* LSTM.predictBiasD) ./ ((LSTM.predictBiasM + 1e-8).^0.5));

end





