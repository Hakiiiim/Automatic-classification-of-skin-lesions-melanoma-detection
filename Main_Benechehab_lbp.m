%%  Automatic classifcation of skin lesions
% ------------------------------------
%   Automatic classifcation of skin lesions
% ------------------------------------
% Abdelhakim Benechehab
clear
clc
close all

%% STEP 1: Organizing Data

% Load the images and labels
truth = readtable("ISIC-2017_Data_GroundTruth_Classification.csv");

%Maximum index of images
N = 519;

%indexes of images from 0 to 518, 1 if corresponding image exist, 0
%otherwise
indexes = zeros(1,N);

%labels of present images : 0 if image not found, 1 if safe lesion, 2 if
%melanoma
labels = zeros(1,N);

k=0;
for i = 1:N
    %Managing errors : the following lines are run only if no error was
    %detected reading the file(the image exist)
    try
        if (i-1) < 10
            name = strcat('ISIC_000000' ,num2str(i-1));
            I = imread(strcat('PROJECT_Data/ISIC_000000' ,num2str(i-1) , '.jpg')) ;
        elseif (i-1) < 100
            name = strcat('ISIC_00000' ,num2str(i-1));
            I = imread(strcat('PROJECT_Data/ISIC_00000' ,num2str(i-1) , '.jpg')) ;
        else
            name = strcat('ISIC_0000' ,num2str(i-1));
            I = imread(strcat('PROJECT_Data/ISIC_0000' ,num2str(i-1) , '.jpg')) ;
        end
        k=1;
    catch
        %in case of corresponding image of 'i' is missing
        warning('fichier inexistant');
    end
    
    if k==1
        %Mark index of image to 1 to show it exists
        indexes(i) = 1;
        
        %refixing the counter to 0
        k=0;
        
        %Check for the image label in the truth table
        for j=1:2001
            if isequal(truth.image_id{j},name)
                labels(i) = 1+truth.melanoma(j);
                break
            end
        end
    end
end

%% Image Descriptors (test)
%test on one image
test = imread('PROJECT_Data/ISIC_0000000.jpg');

R = test(:,:,1);
V = test(:,:,2);
B = test(:,:,3);

RH = lbp(R);
VH = lbp(V);
BH = lbp(B);

H = [RH, VH, BH];

%test of the lbp on the three components
figure
bar(H);
figure
bar(RH);
hold on
bar(VH);
hold on
bar(BH);
legend('Red','Green','Blue');
%% LBP descriptor

LBP = cell(1,519);
%the cot of the lbp funtion is very high, thus we will compute only on the
%R layer for the moment
for i=1:N
    if indexes(i)==1
        if (i-1) < 10
            I = imread(strcat('PROJECT_Data/ISIC_000000' ,num2str(i-1) , '.jpg')) ;
        elseif (i-1) < 100
            I = imread(strcat('PROJECT_Data/ISIC_00000' ,num2str(i-1) , '.jpg')) ;
        else
            I = imread(strcat('PROJECT_Data/ISIC_0000' ,num2str(i-1) , '.jpg')) ;
        end
        
        %R component of image
        R = I(:,:,1);
        
        %LBP histogram
        RH = lbp(R);

        LBP{i} = RH;
        
        disp(i)
    end
end

%% Preparing training data
LBPn = zeros(200,256);
k=1;
for i=1:N
    if indexes(i)==1
        LBPn(k,:) = LBP{i};
        k=k+1;
    end 
end

labelsn = labels(labels>=1);
labelsn = labelsn>=2;

X = LBPn;
Y = labelsn;

%% STEP 2: CLASSIFICATION

% X contains the training patterns (dimension 255)
% Y contains the class label of the patterns (i.e. Y(37) contains the label
% of the pattern X(37,:) ).

% Number of patterns (i.e., elements) and variables per pattern in this
% dataset
[num_patterns, num_features] = size(X);

%%
% Normalization of the data
mu_data = mean(X);
std_data = std(X);
X = (X-mu_data)./std_data;

%%
% Parameter that indicates the percentage of patterns that will be used for
% the training
p_train = 0.75;

%%
% SPLIT DATA INTO TRAINING AND TEST SETS

num_patterns_train = round(p_train*num_patterns);

indx_permutation = randperm(num_patterns);

indxs_train = indx_permutation(1:num_patterns_train);
indxs_test = indx_permutation(num_patterns_train+1:end);

X_train = X(indxs_train, :);
Y_train = Y(indxs_train);

X_test= X(indxs_test, :);
Y_test = Y(indxs_test);

%% Classifiers
classNames = [true,false];

%Decision trees classifier
disp('model_1')
model_1 = fitctree(X_train,Y_train,'OptimizeHyperparameters','auto');
%knn binary classifier
disp('model_2')
model_2 = fitcknn(X_train,Y_train, 'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
%discriminant analysis classifier
disp('model_3')
model_3 = fitcdiscr(X_train,Y_train,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
%naive bayes classifier
disp('model_4')
model_4 = fitcnb(X_train,Y_train,'ClassNames',classNames,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));

%Predictions and probabilities of belonging
[Y_test_asig_1,Y_test_hat_1] = predict(model_1,X_test);
[Y_test_asig_2,Y_test_hat_2] = predict(model_2,X_test);
[Y_test_asig_3,Y_test_hat_3] = predict(model_3,X_test);
[Y_test_asig_4,Y_test_hat_4] = predict(model_4,X_test);

%%
[accuracy1,FScore1] = metrics(Y_test,Y_test_asig_1);
[accuracy2,FScore2] = metrics(Y_test,Y_test_asig_2);
[accuracy3,FScore3] = metrics(Y_test,Y_test_asig_3);
[accuracy4,FScore4] = metrics(Y_test,Y_test_asig_4);

X_cat = categorical({'Tree','KNN','Disc','Bayes'});
Y_met = [accuracy1 FScore1 ; accuracy2 FScore2 ; accuracy3 FScore3 ; accuracy4 FScore4];
figure
bar(X_cat,Y_met)
legend('Accuracy','FScore');

%% PART 2.2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACY AND FSCORE

Y_test_asig = Y_test_asig_3';
Y_test_hat = Y_test_hat_3(:,2);

% Show confusion matrix
figure;
plotconfusion(Y_test, Y_test_asig);

% ACCURACY AND F-SCORE
[accuracy,FScore] = metrics(Y_test,Y_test_asig);

fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100);
fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);

%% Model evaluation: Performance metrics (ROC analysis)

[TPR,FPR] = ROC(Y_test,Y_test_hat);

figure;
plot(FPR,TPR,[0,1],[0,1],'--')
title('Model evaluation: Performance metrics (ROC analysis)');
xlabel('FPR(1-specificity)');
ylabel('TPR(sensitivity)');
legend('ROC Classifier','Random classification');

%The AUC (area under the curve)
[n m] = size(TPR);
q = 0;
%Integral is computed using the rectangle rule
for i=1:(m-1)
    q=q+(FPR(i)-FPR(i+1))*TPR(i);
end

disp(q)











