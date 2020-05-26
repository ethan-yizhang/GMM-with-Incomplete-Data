
function varargout = gmm(X, K_or_centroids,missing_index,missing_numj,max_iters,learing_rate)
% ============================================================
% Expectation-Maximization iteration implementation of
% Gaussian Mixture Model.
%
% PX = GMM(X, K_OR_CENTROIDS)
% [PX MODEL] = GMM(X, K_OR_CENTROIDS)
%
%  - X: N-by-D data matrix.
%  - K_OR_CENTROIDS: either K indicating the number of
%       components or a K-by-D matrix indicating the
%       choosing of the initial K centroids.
%
%  - PX: N-by-K matrix indicating the probability of each
%       component generating each point.
%  - MODEL: a structure containing the parameters for a GMM:
%       MODEL.Miu: a K-by-D matrix.
%       MODEL.Sigma: a D-by-D-by-K matrix.
%       MODEL.Pi: a 1-by-K vector.
%  - missing_index: missing data's indexs
%  - missing_numj: missing features' number
%  - max_iters: max iteration times (default = 100)
% ============================================================

    
 
    %% Generate Initial Centroids

    [N, D] = size(X);
    if isscalar(K_or_centroids) %if K_or_centroid is a 1*1 number
        K = K_or_centroids;
        Rn_index = randperm(N); %random index N samples
        centroids = X(Rn_index(1:K), :); %generate K random centroid
    else % K_or_centroid is a initial K centroid
        K = size(K_or_centroids, 1); 
        centroids = K_or_centroids;
    end
 
    %% initial values
    [pMiu,pPi,pSigma] = init_params();
    threshold = 1e-4;     
    Sigma_shift = 1e-8;
    th = floor(N*0.6); 
    Lprev = -inf; 
    
    %% EM Algorithm
    iters=0;
    maxiters=100;
    if max_iters~=0
        maxiters=max_iters;
    end

    while true
        iters=iters+1;
        %% Estimation Step
        
        log_lh=wdensity(X,pMiu, pSigma, pPi, false, 2);
        [L,pGamma] = estep(log_lh,1e-12);
        obj(iters)= L;


    %% converge judge
        if iters>1 && obj(iters) < obj(iters-1)
               fprintf('%s \n', [ 'error at update data ',num2str(iters),' ',num2str(obj(iters) - obj(iters-1))]);
        end
        if L<Lprev
            fprintf('%s \n', ['error at last ',num2str(iters),' ',num2str(L-Lprev)]);
        end
        if ((L-Lprev)/abs(L) < threshold) || (iters >= maxiters)
            break;
        end
        Lprev = L;
        % pGamma = Px .* repmat(pPi, N, 1); %  numerator:  pi(k) * N(xi | pMiu(k), pSigma(k))
        % pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K); %denominator:    pi(j) * N(xi | pMiu(j), pSigma(j))
 
%% Maximization Step - through Maximize likelihood Estimation
        
        Nk = sum(pGamma, 1); %Nk (1,k)
        Nk(Nk==0) = eps;
        
    %% update miu sigma pri     
        for kk= 1:K
            pGamma_k = pGamma(:,kk);
            nz_idx  = pGamma_k  > 0 ;
            pMiu(kk,:) = pGamma_k' * X / Nk(kk) ;
            if sum(nz_idx) < th %for effiency
                Xshift = X(nz_idx,:)-pMiu(kk, :);
                pGamma_k = pGamma_k(nz_idx) ;
            else
                Xshift = X-repmat(pMiu(kk, :), N, 1);
            end
           
            pSigma(:, :, kk) = (Xshift' * ...
                (diag(pGamma_k) * Xshift)) / (Nk(kk));
             pSigma(:, :, kk)= ( pSigma(:, :, kk)+ pSigma(:, :, kk)')/2+ eye(D)*Sigma_shift;
             
        end
        pPi = Nk/N;

        log_lh=wdensity(X,pMiu, pSigma, pPi, false, 2);
        [~,pGamma] = estep(log_lh,1e-12);
    %% update missing_part_of_data
        learning_rate=1;
        if ~isempty(missing_index)
            for j = 1:N
                num_miss = missing_numj(j);
                if num_miss ~= 0
                    dj_col = X(j,:)';
                    miss_j = missing_index(missing_index(:,1)==j,2); 
                    obs_j = zeros(D-num_miss,1);
                    cnt = 1;
                    for i = 1:D
                        if ismember(i,miss_j)==0
                            obs_j(cnt) = i ;
                            cnt = cnt + 1;
                        end
                    end
                    xo = dj_col(obs_j);
                    xm = dj_col(miss_j);
                    sigu_sum = zeros(num_miss,1);
                    smox_sum = zeros(num_miss,1);
                    smm_sum = zeros(num_miss,num_miss);

                    for i=1:K   %for each components 
                        mu = pMiu(i,:);
                        sigma_org = pSigma(:,:,i);
                        pri = pPi(:,i);
                        mu_col = mu';
                        sigma_inv = inv(sigma_org);
                        smo = sigma_inv(miss_j,obs_j);
                        smm = sigma_inv(miss_j,miss_j);
                        muo = mu_col(obs_j);
                        mum = mu_col(miss_j);                       
                        smm_sum = smm_sum + pGamma(j,i).*smm;
                        sigu_sum = sigu_sum + pGamma(j,i).* (smo*muo + smm*mum);
                        smox_sum = smox_sum + pGamma(j,i).* (smo*xo);
                    end

                   smm_sum = (smm_sum + smm_sum' ) /2;
                   xm_new = (smm_sum ) \ (sigu_sum-smox_sum);
                   xm_new = learning_rate*xm_new + (1-learning_rate)*xm;%learning_rate 
                   X(j,miss_j) = xm_new;

                end
            end
        end
    end
    
   
    if nargout == 1
        varargout = {pGamma};
    else
        model = [];
        model.Miu = pMiu;
        model.Sigma = pSigma;
        model.Pi = pPi;
        varargout = {pGamma, obj,model};
    end
 
    %% Function Definition
    
    function [pMiu,pPi,pSigma] = init_params()
        pMiu = centroids; %k*D,
        pPi = zeros(1, K); %£¨influence factor£©
        pSigma = zeros(D, D, K); %Covariance matrix£¬D*D
 
        % £¨x-pMiu£©^2 = x^2+pMiu^2-2*x*Miu
        distmat = repmat(sum(X.*X, 2), 1, K) + ... %x^2, N*1
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - ...%pMiu^2£¬1*K
            2*X*pMiu';
        [~, labels] = min(distmat, [], 2);%Return the minimum from each row
 
        for k=1:K
            Xk = X(labels == k, :);
            pPi(k) = size(Xk, 1)/N;
            pSigma(:, :, k) = cov(Xk) +Sigma_shift*eye(D);  
        end
    end
 
    
end