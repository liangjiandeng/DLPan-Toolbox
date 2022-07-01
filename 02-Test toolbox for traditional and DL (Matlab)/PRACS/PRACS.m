%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           PRACS fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Partial Replacement Adaptive CS (PRACS) algorithm. 
% 
% Interface:
%           I_Fus_PRACS = PRACS(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_PRACS:    PRACS pansharpened image.
% 
% References:
%           [Choi11]        J. Choi, K. Yu, and Y. Kim, “A new adaptive component-substitution-based satellite image fusion by using partial replacement,” IEEE
%                           Transactions on Geoscience and Remote Sensing, vol. 49, no. 1, pp. 295–309, January 2011.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2019
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_PRACS = PRACS(I_MS,I_PAN,ratio)

beta = 0.95; % for 11-bit data
% beta = 1.95; % for 8-bit data

I_MS = double(I_MS);
I_PAN = double(I_PAN);
[N,M,L] = size(I_MS);

%%% Histogram matching of each MS band to Pan
msexp_hm = zeros(N,M,L);
for k=1:L
    b = I_MS(:,:,k);
    b = (b - mean2(b) + mean2(I_PAN)/std2(I_PAN)*std2(b)) * std2(I_PAN)/std2(b);
    b(b<0) = 0;
    msexp_hm(:,:,k) = b;
end

%%% Computing low-resolution Pan by bicubic decimation/interpolation
aux = imresize(I_PAN,1/ratio);
pan_l = imresize(aux,ratio);
clear aux

%%% Regression of Pan_low vs MS (with offset)
bb = zeros(N*M,L);
for k = 1:L
    bb(:,k) = reshape(squeeze(msexp_hm(:,:,k)),[N*M,1]);
end
bb = [ones(N*M,1),bb];
alpha = regress(pan_l(:),bb); 

%%% Initial estimate of intensity
aux = bb * alpha;
I_l = reshape(aux,[N,M]);
clear aux

clear bb

%%% Partial Replacement
I_h = zeros(N,M,L);
cc  = zeros(1,L);
for k=1:L
    b = msexp_hm(:,:,k);
    cc(k) = corr2(I_l(:),b(:));
    aux = cc(k)*I_PAN(:)+(1-cc(k))*b(:);
    I_h(:,:,k) = reshape(aux,[N,M]);
end
clear aux

%%% Band-dependent intensity
%%% For each band, compute low-resolution I_h by bicubic decimation/interpolation
I_h_low = zeros(N,M,L);
for k=1:L
    aux = imresize(I_h(:,:,k),1/ratio);
    I_h_low(:,:,k) = imresize(aux,ratio);
end
clear aux
%%%

%%% Regression of I_h_low_k vs MS (with offset)

alpha = zeros(L+1,L);
for k = 1:L
    bb(:,k) = reshape(squeeze(msexp_hm(:,:,k)),[N*M,1]);
end
bb = [ones(N*M,1),bb];
for k=1:L
    aux = I_h_low(:,:,k);
    alpha(:,k) = regress(aux(:),bb);
end
clear aux

%%% Intensities
I_l_prime = zeros(N,M,L);
for k=1:L
    aux = bb * alpha(:,k);
    I_l_prime(:,:,k) = reshape(aux,[N,M]);
end
clear aux

%%% Computing detail images
delta = zeros(N,M,L);
for k=1:L
    delta(:,:,k)= I_h(:,:,k)-I_l_prime(:,:,k)-(mean2(I_h(:,:,k))-mean2(I_l_prime(:,:,k)));
end

%%% Computing mean of std. devs.
aux3 = zeros(1,L);
for k=1:L
    aux3(k) = std2(I_MS(:,:,k));
end
aux3 = mean(aux3);

%%% Computing weights
w = zeros(1,L);
for k=1:L
    aux1 = I_l_prime(:,:,k);
    b = I_MS(:,:,k);
    w(k) = beta .* corr2(aux1(:),b(:))*std(b(:))/aux3;%std(aux2(:));
end

%%% Computing local instability adjustment parameter
L_I = zeros(N,M,L);
for k=1:L
    b = I_MS(:,:,k);
    I = I_l_prime(:,:,k);
    aux = 1-abs(1-corr2(I_l(:),b(:))*b(:)./I(:));
    L_I(:,:,k) = reshape(aux,[N,M]);
end

%%% Computing pansharpened image
det = zeros(N,M,L);
I_Fus_PRACS = zeros(N,M,L);
for k=1:L
    det(:,:,k) = w(k) * L_I(:,:,k) .* delta(:,:,k);
    I_Fus_PRACS(:,:,k) = I_MS(:,:,k) + det(:,:,k);
end

end