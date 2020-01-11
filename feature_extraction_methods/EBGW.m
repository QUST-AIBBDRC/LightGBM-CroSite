clear all
clc
[data,A,B]=xlsread('data_LYN.xlsx','pos');
AA=A(2:end,4);
%input=importdata('set1_positive.txt');
%data=input(1:end,:);
[m,n]=size(AA);
vector=[];
for i=1:m;
 vector= [vector;EBGW_demo(AA{i})];
 end
save EBGW_pos_31.mat vector


