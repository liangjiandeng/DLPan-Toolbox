%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           C_BDSD fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images  
%           through the Clustered Band-Dependent Spatial-Detail (C-BDSD) algorithm. 
% 
% Interface:
%           I_Fus_C_BDSD = C_BDSD(I_MS,I_PAN,ratio,sensor,K)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS').
%           K:              Number of clusters (K>1) (Optional: default value K=30); 
%
% Outputs:
%           I_Fus_C_BDSD:   C_BDSD pansharpened image.
% 
% Reference:
%           [Garzelli15]    A. Garzelli, “Pansharpening of Multispectral Images Based on Nonlocal Parameter Optimization,” 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 4, pp. 2096-2107, April 2015.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_C_BDSD = C_BDSD(I_MS,I_PAN,ratio,sensor,K)

%%%
% Control of input parameters and initialization
%%%
[N,M,Nb] = size(I_MS);

if nargin == 5
    if K < 2
        fprintf(1,'Required number of clusters K>1.\n\n'); 
    return
    end
end
if nargin < 5
    K = 30; 
end
if nargin < 4
    fprintf(1,'\nI_Fus_C_BDSD = C_BDSD(I_MS,I_PAN,ratio,sensor,K)\n\n');
    error('At least four input arguments required')
end

I_MS = double(I_MS);
I_PAN = double(I_PAN);

%%%
% Reduced resolution
%%%

pan_LP = MTF_PAN(I_PAN,sensor,ratio);
pan_LP_d = pan_LP(3:ratio:end,3:ratio:end);

ms_orig = imresize(I_MS,1/ratio);
ms_LP_d = MTF(ms_orig,sensor,ratio);


% CLUSTER MAPS AT FULL RESOLUTION AND REDUCED RESOLUTION
%
Sa = stdfilt(I_PAN,ones(51)); 
Sa = Sa/max(Sa(:));
Sb = I_PAN; 
Sb = Sb/max(Sb(:));
 
opts = statset('TolX',1e-5);

features = zeros(N/ratio,M/ratio,2,ratio*ratio);
for i = 1:ratio
    for j = 1:ratio
        features(:,:,1,(i-1)*ratio+j) = Sa(1+(i-1):ratio:end,1+(j-1):ratio:end);
        features(:,:,2,(i-1)*ratio+j) = Sb(1+(i-1):ratio:end,1+(j-1):ratio:end);
    end
end
C_stack = zeros(N/ratio,M/ratio,ratio*ratio);

f = features(:,:,:,(3-1)*ratio+3);
warning off
[aux, centers] = kmeans(reshape(f,[N/ratio*M/ratio,2]),K,'replicates',2,'start','cluster','options',opts);
C = reshape(aux,[N/ratio M/ratio]);
C_stack(:,:,(3-1)*ratio+3) = C;
for i = 1:ratio
    for j = 1:ratio
        if(i*j~=9)
            f = features(:,:,:,(i-1)*ratio+j);
            aux = kmeans(reshape(f,[N/ratio*M/ratio,2]),K,'start',centers,'MaxIter',1);
            C_stack(:,:,(i-1)*ratio+j) = reshape(aux,[N/ratio M/ratio]);
        end
    end
end

C4 = zeros(size(I_PAN));
for i = 1:ratio
    for j = 1:ratio
        C4(i:ratio:end,j:ratio:end) = C_stack(:,:,(i-1)*ratio+j);
    end
end

% ESTIMATE PARAMETERS AT REDUCED RESOLUTION AND INJECT (CLUSTER BY CLUSTER)
%
g = zeros(K,Nb);
alpha = zeros(Nb,Nb,K);
offset = zeros(Nb,K);
ms_ps_stack = zeros(N,M,Nb,K);

% Estimate for K=1
[~,g_global,alpha_global,offset_global] = parm_est(ms_LP_d(:,:,:),pan_LP_d,ms_orig,find(C>0));

for j=1:K
    [~,g(j,:),alpha(:,:,j),offset(:,j)] = parm_est(ms_LP_d(:,:,:),pan_LP_d,ms_orig,find(C==j));
    if(size(find(g<0)>0))
        g(j,:) = g_global;
        alpha(:,:,j) = alpha_global;
        offset(:,j) = offset_global;
    end
    H = H_comp(I_PAN,I_MS,find(C4==j));
    ms_ps_stack(:,:,:,j) = bdsd_injection(I_PAN,I_MS,H,g(j,:),squeeze(alpha(:,:,j)),offset(:,j),find(C4==j));
end

% FORM PANSHARPENED IMAGE
I_Fus_C_BDSD = sum(ms_ps_stack,4);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [gamma,g,alpha,offset] = parm_est(hs_LP_d,pan_LP_d,hs_orig,ind)

Nb = size(hs_orig,3);

for i=1:Nb
    % compute Hd
    Hd = zeros(size(ind,1),Nb+2);
    gamma = zeros((Nb+2),Nb);
    for k=1:Nb
        bfull = hs_LP_d(:,:,k);
        Hd(:,k) = bfull(ind);
    end
    Hd(:,Nb+1) = ones(size(ind));
    Hd(:,Nb+2) = pan_LP_d(ind);
    
    % estimate gamma
    
    for k=1:Nb
        Z = (Hd'*Hd)\Hd';
        bfull = hs_orig(:,:,k);
        b = bfull(ind);
        bdfull = hs_LP_d(:,:,k);
        bd = bdfull(ind);
        gamma(:,k) = Z *(b(:)-bd(:));
    end
    
    g = gamma(Nb+2,:);
    
    alpha = zeros(Nb);
    for k = 1:Nb
        alpha(:,k) = -gamma(1:Nb,k)/gamma(Nb+2,k);
    end
    
    offset = zeros(Nb,1);
    for k = 1:Nb
        offset(k) = gamma(Nb+1,k)/gamma(Nb+2,k);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function hs_en = bdsd_injection(pan,msexp,H,g,alpha,offset,ind)

[N,M,Nb] = size(msexp);
Intensity = zeros(length(ind),Nb);
for k = 1:Nb
    Intensity(:,k) = H(:,1:Nb) * alpha(:,k) - offset(k);
end
pfull = pan;
p = pfull(ind);

hs_en = zeros(N,M,Nb); 
for k=1:Nb
    bfull = msexp(:,:,k);
    b = bfull(ind);
    b_en = b(:) + (p - Intensity(:,k)) * g(k);
    hs_enfull = hs_en(:,:,k);
    hs_enfull(ind) = b_en;
    hs_en(:,:,k) = reshape(hs_enfull,N,M);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function H = H_comp(pan,hs,ind)

Nb = size(hs,3);
H = zeros(length(ind),Nb+2);

for k=1:Nb
    bfull = hs(:,:,k);
    H(:,k) = bfull(ind);
end
H(:,Nb+1) = ones(size(ind));
H(:,Nb+2) = pan(ind);

