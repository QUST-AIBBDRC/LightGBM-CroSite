function [  pwaa] = PWAA_demo( protein )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% clear all
% clc
% protein='TFVPAKKVKDLLK';
input=protein;
L = length(input);
h=[];
inp = zeros(1,L);
for ii = 1:L;
    if strcmpi(input(ii),'A')== 1  %Ala
        inp(ii) = 1;
    elseif strcmpi(input(ii),'C')== 1  %Cys
        inp(ii) = 2; 
    elseif strcmpi(input(ii),'D')== 1  %Asp
        inp(ii) = 3;
    elseif strcmpi(input(ii),'E')== 1  %Glu
        inp(ii) = 4;
    elseif strcmpi(input(ii),'F')== 1  %Phe
        inp(ii) = 5;
    elseif strcmpi(input(ii),'G')== 1  %Gly
        inp(ii) = 6;
    elseif strcmpi(input(ii),'H')== 1  %His
        inp(ii) = 7;
    elseif strcmpi(input(ii),'I')== 1  %Ile
        inp(ii) = 8;
    elseif strcmpi(input(ii),'K')== 1  %Lys
        inp(ii) = 9;
    elseif strcmpi(input(ii),'L')== 1  %Leu
        inp(ii) = 10;
    elseif strcmpi(input(ii),'M')== 1  %Met
        inp(ii) = 11;
    elseif strcmpi(input(ii),'N')== 1  %Asn
        inp(ii) = 12;
    elseif strcmpi(input(ii),'P')== 1  %Pro
        inp(ii) = 13;
    elseif strcmpi(input(ii),'Q')== 1  %Gln
        inp(ii) = 14;
    elseif strcmpi(input(ii),'R')== 1  %Arg
        inp(ii) = 15;
    elseif strcmpi(input(ii),'S')== 1  %Ser
        inp(ii) = 16;
    elseif strcmpi(input(ii),'T')== 1  %Thr
        inp(ii) = 17;
    elseif strcmpi(input(ii),'V')== 1  %Val
        inp(ii) = 18;
    elseif strcmpi(input(ii),'W')== 1  %Trp
        inp(ii) = 19;
    elseif strcmpi(input(ii),'Y')== 1  %Tyr
        inp(ii) = 20;
    else
        inp(ii) = 21;  %undefined
    end
end
ll=(L-1)/2;
x=zeros(21,L);

vector=zeros(1,21);
for i=1:21;
  for ii=1:L;
     if (inp(ii)==i);
        x(i,ii)=1;
%         for j=-ll:ll;
%            vector(i,j)=1/(ll*(ll+1))*sum(x(i,ll)*(j+abs(j)/l));
%         end
     end
   end
end
a=1:L;
b=a-ll-1;
d=abs(a-ll-1)/ll;
for i=1:21;
    j=-ll:ll;
    pwaa(i)=1/(ll*(ll+1))*sum((x(i,j+ll+1).*(j+abs(j)/ll)));
end


end

