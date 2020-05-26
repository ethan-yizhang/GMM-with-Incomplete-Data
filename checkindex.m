function [index,num_missingj] = checkindex(data,indexnum)

sample = size(data,1);
dimension = size(data,2);

index = zeros(indexnum,2);
num_missingj = zeros(sample,1);
c = 0;
init = 1;
for i = 1:sample
    nm=0;
    for j = 1:dimension
        if isnan(data(i,j)) == 1
            index(init,1) = i;
            index(init,2) = j;
            init = init + 1;
            c = c+1;
            nm = nm+1;
        end
    end
    num_missingj(i) = nm;
end


end



