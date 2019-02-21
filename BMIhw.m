%% BMI Homework
clear all; close all
clc
load('monkeydata_training.mat')
trial1 = trial(1,1);

%%
% each bin (increment) is 1ms
fs = 1000;

% rasterplot.m - open source, credit Rajiv Narayan, Boston U, sourced from:
% https://uk.mathworks.com/matlabcentral/fileexchange/10000-rasterplot

% rasterplot for a single trial, all neurons

rasterplot(find(trial1.spikes), length(trial1.spikes(:,1)), length(trial1.spikes(1,:)))

% rasterplot for a single neuron, multiple trials
% choose one arm trajectory

p0Trials = trial(:,1);

l = 1000000000;
for i = 1:length(p0Trials)
   if(length(p0Trials(i).spikes(1,:)) < l)
       l = length(p0Trials(i).spikes(1,:));
   end
  
end

for i = 1:length(p0Trials)
    n1Spikes(i,1:l) = p0Trials(i).spikes(1,1:l); 
end

rasterplot(find(n1Spikes), length(n1Spikes(:,1)), length(n1Spikes(1,:)))

%% PSTH for neuron 1, trajectory 0

timeStep = 5; % in samples
t = 1:timeStep:l;

for i = 1:length(t)-1
    PSTH(i) = (1/timeStep).*1/length(n1Spikes(:,1)) * sum(n1Spikes(:,t(i):t(i+1)),'all');
end

%histogram
bar(t(1:end-1),PSTH)
set(gca,'FontSize',28)
xlabel('Samples','Interpreter','Latex','fontsize',35)
ylabel('PSTH','Interpreter','Latex','fontsize',35)
grid on

figure
%smoothed plot
plot(t(1:end-1),filter((1/timeStep)*ones(1,timeStep),1,PSTH))
set(gca,'FontSize',28)
xlabel('Samples','Interpreter','Latex','fontsize',35)
ylabel('PSTH (Moving Average)','Interpreter','Latex','fontsize',35)
grid on

%PSTH(find(n1Spikes), 10, 1000, 100, l)

%% Hand position plotting

% choose 8 trials, one from each reaching angle
% plot hand positions

%3Dclos
figure
for i = 1:8
    plot3(trial(1,i).handPos(1,:),trial(1,i).handPos(2,:),trial(1,i).handPos(3,:));
    set(gca,'FontSize',28)
    xlabel('X','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    ylabel('Y','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    zlabel('Z','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    grid on
    hold on 
end

%2D
figure
for i = 1:8
    plot(trial(1,i).handPos(1,:),trial(1,i).handPos(2,:));
    set(gca,'FontSize',28)
    xlabel('X','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    ylabel('Y','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    grid on
    hold on 
end

% Compare Variance within single angle (0) for 10 trials 
figure
for i = 1:10
    plot(trial(i,1).handPos(1,:),trial(i,1).handPos(2,:));
    set(gca,'FontSize',28)
    xlabel('X','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    ylabel('Y','Interpreter','Latex','fontsize',35, 'lineWidth',7)
    grid on
    hold on 
end

%% Plot tuning curves for 8 angles
% Choose neurons 1 through 10



radAxis = [30*pi/180, 70*pi/180, 110*pi/180, 150*pi/180, 190*pi/180, 230*pi/180, 310*pi/180, 350*pi/180];
% find time average of spiking for each directon - neuron 1
for neuron = 1:5
totalRate = 0;
totalVar = 0;
    for i = 1:8
        for j =  1:100
            totalRate = totalRate + mean(trial(j,i).spikes(neuron,:));
            totalVar = totalVar + var(trial(j,i).spikes(neuron,:));
        end
        meanRate(i) = totalRate/100;
        meanVar(i) = totalVar/100;
        totalRate = 0;
        totalVar = 0;
    end
    
    % circular plots - "Circular Statistics Toolbox" - not used but useful
    % resource
    % https://www.jstatsoft.org/article/view/v031i10
    

    weightedAxis = (radAxis.*meanRate);
    [~, maxAxis] = max(meanRate);
    
    maxAxis = radAxis(maxAxis);
    
    meanMag = mean(meanRate);
    
    % Need to plot std first for scaling
    figure
    polarplot([radAxis radAxis(1)],[(meanRate + 0.1.*sqrt(meanVar)) meanRate(1)+0.1.*sqrt(meanVar(1))],'o--')
    hold on
    polarplot([radAxis radAxis(1)],[meanRate meanRate(1)])
    polarplot([radAxis radAxis(1)],[(meanRate - 0.1.*sqrt(meanVar)) meanRate(1)-0.1.*sqrt(meanVar(1))],'o--')
    polarplot([0 sum(weightedAxis)], [0 meanMag],'g-')
    % polarplot([0 median(weightedAxis)], [0 median(meanRate)],'m')
    polarplot([0 maxAxis], [0 max(meanRate)],'-k', 'LineWidth',2)
    legend('Response + 0.1 std','Response','Response - 0.1 std','Mean Response','Modal Response')
    hold off
    
    
end

%% Population Vector - Find preferred direction of each neuron
for neuron = 1:98
totalRate = 0;
totalVar = 0;
    for i = 1:8
        for j =  1:100
            totalRate = totalRate + mean(trial(j,i).spikes(neuron,:));
        end
        rates(i) = totalRate;
        totalRate = 0;
        totalVar = 0;
    end
    [~, idx] = max(rates);
    pDirect(neuron) = idx;
end

