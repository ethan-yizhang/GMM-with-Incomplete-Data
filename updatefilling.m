function [ data ] = updatefilling(data,U,labels_u,index)

[num,~]=size(index);
for i=1:num
    sampleindex = index(i,1);
    dimensionindex = index(i,2);
    label = labels_u(sampleindex);
    data(sampleindex,dimensionindex) = U(label,dimensionindex);
end 
end

