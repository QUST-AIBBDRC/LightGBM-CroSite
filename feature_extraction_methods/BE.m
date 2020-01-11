clear all
clc
%data=load('data_LYN.xlsx');
[data,A,B]=xlsread('data_LYN.xlsx','pos');
AA=A(2:end,4);
%data_pos=AA(:,4);
[m,n]=size(AA);
sequence=AA;
sequence_label=[ones(m,1)];
num=m;
output1=[];
input=sequence;
for i=1:num
    protein=input{i};
    output =exchange_matrix(protein);
    output1=[output1;output];
end
matrix=output1;
[num1,column]=size(matrix);
feature=[];
vector=[];
for i=1:num1
    A=matrix(i,:);
    for j=1:column
        feature=zeros(1,21);
        feature(A(j))=1;
        vector=[vector,feature];
        feature=[];
    end
    vec{i,1}=vector;
    vector=[];
    A=[];
end
matrix=cell2mat(vec);
save BE_pos.mat matrix