%% 计算聚类效果熵与纯度 输入的矩阵为 CP：算法聚类与实际类别得到的数据的交集
%Ci 算法聚类得到的每个类别的总数
function [Entropy,Purity]=EnAndPur(CP,Ci)
%得到行列值
[rn, cn]=size(CP);
%% 计算熵
% %计算概率 precision
Entropy=0;
precision=zeros(rn,cn);
for i=1:rn
    for j=1:cn
     precision(i,j)=CP(i,j)/Ci(1,i);    
    end
end
% %计算ei(i,j)
% for i=1:rn
%     for j=1:cn
%      ei(i,j)=precision(i,j)*log2(precision(i,j));    
%     end
% end
% %
% %计算ei_sum
% for i=1:rn
%     ei_sum(i)=-nansum(ei(i,:));
% end
% %计算mi*ei_sum(i)
% for j=1:cn
%     mmi(j)=Ci(1,j)*ei_sum(j);
% end
% %计算entropy
% Entropy=nansum(mmi)/nansum(Ci);
%% 计算纯度Purity
%找出最大的一类
pr_max=zeros(1,rn);
for i=1:rn
     pr_max(i)=max(precision(i,:));    
end
%计算类别数量
nni=zeros(1,rn);
for j=1:cn
    nni(j)=Ci(1,j)*pr_max(j);
end
Purity=nansum(nni)/nansum(Ci);

end