clc
clear
loadpath = '*';
resultpath = '*';

dataset={ '*' };
data_nums=length(dataset);
rdtimes_all=10;
for datath =  1:data_nums
    dataname = char(dataset(datath));
    fprintf('%s \n',dataname);
    originmatrix = load([loadpath,'/',dataname,'/',dataname,'_dataset.mat']);
    targets = originmatrix.([dataname,'Targets']);
    data = originmatrix.([dataname,'Inputs']);
    data=data';
    targets = double(targets);
    truelabel = targets(1,:)';
    ratio = [0:10:70];
    
    
    classnumber = length(unique(targets));
    times = 50;
    
    
    
    accval1 = zeros(times,1);
    nmival1 = zeros(times,1);
    Fmval1 = zeros(times,1);
    purval1 = zeros(times,1);
    accval2 = zeros(times,1);
    nmival2 = zeros(times,1);
    Fmval2 = zeros(times,1);
    purval2 = zeros(times,1);
    accval3 = zeros(times,1);
    nmival3 = zeros(times,1);
    Fmval3 = zeros(times,1);
    purval3 = zeros(times,1);
    accval4 = zeros(times,1);
    nmival4 = zeros(times,1);
    Fmval4 = zeros(times,1);
    purval4 = zeros(times,1);
    accval5 = zeros(times,1);
    nmival5 = zeros(times,1);
    Fmval5 = zeros(times,1);
    purval5 = zeros(times,1);
    accval6 = zeros(times,1);
    nmival6 = zeros(times,1);
    Fmval6 = zeros(times,1);
    purval6 = zeros(times,1);
    accval7 = zeros(times,1);
    nmival7 = zeros(times,1);
    Fmval7 = zeros(times,1);
    purval7 = zeros(times,1);
    accval8 = zeros(times,1);
    nmival8 = zeros(times,1);
    Fmval8 = zeros(times,1);
    purval8 = zeros(times,1);
    res = zeros(4,13);
    
    for randtimes = 1:rdtimes_all
        for m=1:length(ratio)
            data = load([loadpath,'/',dataname,'/',dataname,'_missing_',num2str(ratio(m)),'_',num2str(randtimes),'.mat']);
            data = data.absentmatrix;
            data = double(data);
            sample = size(data,1);
            dimension =size(data,2);
            indexnum = ceil(sample*dimension*ratio(m)/100);
            [missingindex,missing_numj] = checkindex(data,indexnum);
            if ratio(m)==0
                missingindex=[];
            end
            empty_missingindex = [];
            empty_missing_numj = zeros(sample,1);
            parfor i =1:times                
                fprintf('%s \n',[dataname '_missing_' num2str(ratio(m)) '_times_' num2str(i)]);
                data_mean = meanfilling(data,missingindex);
                datanew = standardizematrix(data);
                data_zero = zerofilling(datanew,missingindex);
                data_em =  DataCompletion(datanew,'EM');
                                
                %%GMM with incomplete data
                datanew = data_em;
                [~,centroids]=mykmeans(datanew, classnumber,'Replicates',1,'Start','sample','rdseed',i);
                [pGamma,~,~ ] = gmm(datanew, centroids,missingindex,missing_numj,500,truelabel);
                [~,assignment]=max(pGamma,[],2);
                res1 = myNMIACC(assignment,truelabel);
                accval1(i,1) = res1(1);
                nmival1(i,1)= res1(2);
                Fmval1(i,1) = res1(3);
                purval1(i,1) = res1(4);
                
                %%GMM+Mean
                datanew = data_mean;
                [~,centroids]=mykmeans(datanew, classnumber,'Replicates',1,'Start','sample','rdseed',i);
                [pGamma,~,~] = gmm(datanew, centroids,empty_missingindex,empty_missing_numj,500,truelabel);
                [~,assignment]=max(pGamma,[],2);
                res1 = myNMIACC(assignment,truelabel);
                accval2(i,1) = res1(1);
                nmival2(i,1)= res1(2);
                Fmval2(i,1) = res1(3);
                purval2(i,1) = res1(4);
                
                %%GMM+Zero
                datanew = data_zero;
                [~,centroids]=mykmeans(datanew, classnumber,'Replicates',1,'Start','sample','rdseed',i);
                [pGamma,~,~ ] = gmm(datanew, centroids,empty_missingindex,empty_missing_numj,500,truelabel);
                [~,assignment]=max(pGamma,[],2);
                res1 = myNMIACC(assignment,truelabel);
                accval3(i,1) = res1(1);
                nmival4(i,1)= res1(2);
                Fmval3(i,1) = res1(3);
                purval3(i,1) = res1(4);
                
                %%GMM+Em
                datanew = data_em;
                [~,centroids]=mykmeans(datanew, classnumber,'Replicates',1,'Start','sample','rdseed',i);
                [pGamma,~,~, ] = gmm(datanew, centroids,empty_missingindex,empty_missing_numj,500,truelabel);
                [~,assignment]=max(pGamma,[],2);
                res1 = myNMIACC(assignment,truelabel);
                accval4(i,1) = res1(1);
                nmival4(i,1)= res1(2);
                Fmval4(i,1) = res1(3);
                purval4(i,1) = res1(4);
                
                %%DK+Mean
                datanew = data_mean;
                [~,assignment,~] = kmeansfilling(datanew,classnumber,missingindex,i);
                res1 = myNMIACC(assignment,truelabel);
                accval5(i,1) = res1(1);
                nmival5(i,1)= res1(2);
                Fmval5(i,1) = res1(3);
                purval5(i,1) = res1(4);
                
                %%DK+Zero
                datanew = data_zero;
                [~,assignment,~] = kmeansfilling(datanew,classnumber,missingindex,i);
                res1 = myNMIACC(assignment,truelabel);
                accval6(i,1) = res1(1);
                nmival6(i,1)= res1(2);
                Fmval6(i,1) = res1(3);
                purval6(i,1) = res1(4);
                
                %%DK+Em
                datanew = data_em;
                [~,assignment,~] = kmeansfilling(datanew,classnumber,missingindex,i);
                res1 = myNMIACC(assignment,truelabel);
                accval7(i,1) = res1(1);
                nmival7(i,1)= res1(2);
                Fmval7(i,1) = res1(3);
                purval7(i,1) = res1(4);   
            end
                  
            res(1,1) = mean(accval1);
            res(2,1) = mean(nmival1);
            res(3,1) = mean(Fmval1);
            res(4,1) = mean(purval1);
            res(1,2) = mean(accval2);
            res(2,2) = mean(nmival2);
            res(3,2) = mean(Fmval2);
            res(4,2) = mean(purval2);
            res(1,3) = mean(accval3);
            res(2,3) = mean(nmival3);
            res(3,3) = mean(Fmval3);
            res(4,3) = mean(purval3);
            res(1,4) = mean(accval4);
            res(2,4) = mean(nmival4);
            res(3,4) = mean(Fmval4);
            res(4,4) = mean(purval4);
            res(1,5) = mean(accval5);
            res(2,5) = mean(nmival5);
            res(3,5) = mean(Fmval5);
            res(4,5) = mean(purval5);
            res(1,6) = mean(accval6);
            res(2,6) = mean(nmival6);
            res(3,6) = mean(Fmval6);
            res(4,6) = mean(purval6);
            res(1,7) = mean(accval7);
            res(2,7) = mean(nmival7);
            res(3,7) = mean(Fmval7);
            res(4,7) = mean(purval7);            
            allres=[accval1 accval2 accval3 accval4 accval5 accval6 accval7 ...
                nmival1 nmival2 nmival3 nmival4 nmival5 nmival6 nmival7...
                Fmval1 Fmval2 Fmval3 Fmval4 Fmval5 Fmval6 Fmval7 Fmval8...
                purval1 purval2 purval3 purval4 purval5 purval6 purval7];          
            save([resultpath,'/',dataname,'/',dataname,'_missing',num2str(ratio(m)),'_rt',num2str(randtimes),'_all.mat'],'allres')
            save([resultpath,'/',dataname,'/',dataname,'_missing',num2str(ratio(m)),'_rt',num2str(randtimes),'.mat'],'res')
        end
               
    end

end
