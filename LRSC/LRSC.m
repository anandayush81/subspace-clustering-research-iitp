clc, clear all, close all
data = dlmread('liver.arff',',');
x=data(:,1:end-1);
[xn,normx]= cnormalize(x);


