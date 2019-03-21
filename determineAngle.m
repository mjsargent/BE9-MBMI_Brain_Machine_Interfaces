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
    for i = 1:20
       XTest{angle,i} = trial(i+80,angle).spikes;
       YTest{angle,i} = trial(i+80,angle).handPos(1:2,:);
    end
end

timeStep = 20;

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

results = zeros(8, 20);

timeSteps = 20;

for currentAngle = 1:8
    for currentTrial = 1:length(XTestCount(angle,:))
        nearestAngle = getNearestAngle(XTrainCount, XTestCount(currentAngle, currentTrial), timeSteps);
        results(currentAngle, currentTrial) = currentAngle - nearestAngle;
    end
end

function angle = getNearestAngle(XTrainCount, testExample, timeSteps)
    bestMin = realmax;
    testExample = testExample{1,1};
    for currentAngle = 1:8
        for currentTrial = 1:length(XTrainCount(currentAngle,:))
            currentTrainExample = XTrainCount(currentAngle, currentTrial);
            currentTrainExample = currentTrainExample{1,1};
            
            timesToCheck = min(size(testExample, 2), size(currentTrainExample, 2));
            
            timeSteps = min(timeSteps, timesToCheck);

            angles = zeros(1, timesToCheck);
            
            testSlice = testExample(:, 1:timeSteps);
            trainSlice = currentTrainExample(:, 1:timeSteps);

            distance = trainSlice - testSlice;
            distance = sum(distance.^2, 'all')^0.5;
            
            if distance < bestMin
                bestMin = distance;
                angle = currentAngle;
            end
        end
    end
end
