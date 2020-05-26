function RI = RandIndex(X,Y)
%RANDINDEX X&Y are two vectors representing the label of the dataset.
% X input; Y output.
 
X = X(:);
Y = Y(:);
 
TP = 0;
TN = 0;
N = length(X);
C_N2 = nchoosek(N,2);
PIS = [1:N];                                            % PIS: Positive Integer Sequence
ordinal_sequence = nchoosek(PIS,2);
for k = 1:C_N2
    index = ordinal_sequence(k,:);
    if (X(index(1))==X(index(2)) && Y(index(1))==Y(index(2)))
        TP = TP+1;
    end
    if (X(index(1))~=X(index(2)) && Y(index(1))~=Y(index(2)))
        TN = TN+1;
    end
end
RI = (TP+TN)/C_N2;  