%--------------------------------------------------------------------------
% This is the main function for running SSC. 
% Load the DxN matrix X representing N data points in the D dim. space 
% living in a union of n low-dim. subspaces.
% The projection step onto the r-dimensional space is arbitrary and can 
% be skipped. In the case of using projection there are different types of 
% projections possible: 'NormalProj', 'BernoulliProj', 'PCA'. Please refer 
% to DataProjection.m for more information.
%--------------------------------------------------------------------------
% X: DxN matrix of N points in D-dim. space living in n low-dim. subspaces
% s: groundtruth for the segmentation
% n: number of subspaces
% r: dimension of the projection e.g. r = d*n (d: max subspace dim.)
% Cst: 1 if using the constraint sum(c)=1 in Lasso, else 0
% OptM: optimization method {'L1Perfect','L1Noise','Lasso','L1ED'}, see 
% SparseCoefRecovery.m for more information
% lambda: regularization parameter for 'Lasso' typically in [0.001,0.01] 
% or the noise level for 'L1Noise'. See SparseCoefRecovery.m for more 
% information.
% K: number of largest coefficients to pick in order to build the
% similarity graph, typically K = max{subspace dimensions} 
% Missrate: vector of misclassification rates
%--------------------------------------------------------------------------
% In order to run the code CVX package must be installed in Matlab. It can 
% be downlaoded from http://cvxr.com/cvx/download
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------

clc, clear all, close all
data = dlmread('liver.arff',',');
[row, col]= size(data);
class= data(:,end);
C = unique(class);
%data=cnormalize(data);
A= zeros(numel(C),row,col);
for m =1:numel(C)
    for p=1:row
        if data(p,col)==C(m)
            A(m,p,:)= data(p,:);
        end
    end
end
c1= squeeze(A(1,:,:));
c2= squeeze(A(2,:,:));
% c3= squeeze(A(3,:,:));
% c4= squeeze(A(4,:,:));
% c5= squeeze(A(5,:,:));
% c6= squeeze(A(6,:,:));
% c7= squeeze(A(7,:,:));
% c8= squeeze(A(8,:,:));
% c9= squeeze(A(9,:,:));
% c10= squeeze(A(10,:,:));
% c11= squeeze(A(11,:,:));

c1( all(~c1,2), : ) = [];
c2( all(~c2,2), : ) = [];
% c3( all(~c3,2), : ) = [];
% c4( all(~c4,2), : ) = [];
% c5( all(~c5,2), : ) = [];
% c6( all(~c6,2), : ) = [];
% c7( all(~c7,2), : ) = [];
% c8( all(~c8,2), : ) = [];
% c9( all(~c9,2), : ) = [];
% c10( all(~c10,2), : ) = [];
% c11( all(~c11,2), : ) = [];
%dataset=data(:,1:end-1);

%D = 30; %Dimension of ambient space
D=6;
%n = 9; %Number of subspaces
n=numel(C);
d1 = 6; d2 = 6; %d1 and d2: dimension of subspace 1 and 2
%N1 = 20; N2 = 20; %N1 and N2: number of points in subspace 1 and 2
%N1=151; N2=47;

%X1 = randn(D,d1) * randn(d1,N1); %Generating N1 points in a d1 dim. subspace
%X2 = randn(D,d2) * randn(d2,N2); %Generating N2 points in a d2 dim. subspace
e1=c1(:,1:end-1);
e2=c2(:,1:end-1);
% e3=c3(:,1:end-1);
% e4=c4(:,1:end-1);
% e5=c5(:,1:end-1);
% e6=c6(:,1:end-1);
% e7=c7(:,1:end-1);
% e8=c8(:,1:end-1);
% e9=c9(:,1:end-1);
% e10=c10(:,1:end-1);
% e11=c11(:,1:end-1);

X1=e1';
X2=e2';
% X3=e3';
% X4=e4';
% X5=e5';
% X6=e6';
% X7=e7';
% X8=e8';
% X9=e9';
% X10=e10';
% X11=e11';

X = [X1 X2 ]; % X3 X4 X5 X6 X7 X8 X9 X10 X11
%s = [1*ones(1,N1) 2*ones(1,N2)]; %Generating the ground-truth for evaluating clustering results
s=class;


r = 0; %Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
Cst = 0; %Enter 1 to use the additional affine constraint sum(c) == 1
OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
lambda = 0.001; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
K = max(d1,d2); %Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients
if Cst == 1
    K = max(d1,d2) + 1; %For affine subspaces, the number of coefficients to pick is dimension + 1 
end

Xp = DataProjection(X,r,'NormalProj');
CMat = SparseCoefRecovery(Xp,Cst,OptM,lambda);
[CMatC,sc,OutlierIndx,Fail] = OutlierDetection(CMat,s);
if (Fail == 0)
    CKSym = BuildAdjacency(CMatC,K);
    Grps = SpectralClustering(CKSym,n);
    Grps = bestMap(sc,Grps);
    Missrate = sum(sc(:) ~= Grps(:)) / length(sc);
    %disp(Missrate)
    
    save('Lasso_001.mat', 'CMat', 'CKSym','Missrate' ,'Fail', 'Grps');
else
    save('Lasso_001.mat','CMat','Fail');
end