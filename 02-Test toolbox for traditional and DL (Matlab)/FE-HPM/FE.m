%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           FE estimates the estraction detail filter via deconvolution.  
% 
% Interface:
%           PSF_l = FE(I_MS,I_PAN,ratio,tap,lambda,mu,th,num_iter,filtername)
%
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%           tap:                Filter support;
%           lambda:             Coefficient for weighting the energy regularization term;
%           mu:                 Coefficient for weighting the derivative regularization terms;
%           th:                 Threshold on the kernel (it cuts to 0 values below threshold);
%           num_iter:           Max number of iteration (at least 3; not sensitive);    
%           filtername:         Kind of derivative (default: 'Basic')       
%
% Output:
%           PSF_l:              Estimated point spread function.
% 
% Reference:
%           [Vivone15]      G. Vivone, M. Simoes, M. Dalla Mura, R. Restaino, J. Bioucas-Dias, G. A. Licciardi, and J. Chanussot, "Pansharpening based on semiblind deconvolution", 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 4, pp. 1997-2010, 2015.
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
function PSF_l = FE(I_MS,I_PAN,ratio,tap,lambda,mu,th,num_iter,filtername)

if rem(tap,2) == 0
    sum_tap = 0;
else
    sum_tap = 1;
end
tap = floor(tap/2);

[R_SIZE,C_SIZE] = size(I_PAN);

switch filtername
    case 'Naive2'
        gv = zeros(2,1);
        gv(1,1) = -1;
        gv(2,1) = 1;
        
        gh = zeros(1,2);
        gh(1,1) = -1;
        gh(1,2) = 1;
    case 'Naive3'
        gv = zeros(3,1);
        gv(1,1) = -1;
        gv(3,1) = 1;
        
        gh = zeros(1,3);
        gh(1,1) = -1;
        gh(1,3) = 1;
    case 'Basic'
        gv = zeros(2,2);
        gv(1,:) = -1;
        gv(2,:) = 1;
        
        gh = zeros(2,2);
        gh(:,1) = -1;
        gh(:,2) = 1;
    case 'Prewitt'
        gv = zeros(3,3);
        gv(1,:) = -1;
        gv(3,:) = 1;
        
        gh = zeros(3,3);
        gh(:,1) = -1;
        gh(:,3) = 1;
    case 'Sobel'
        gv = zeros(3,3);
        gv(1,1) = -1;gv(1,2) = -2;gv(1,3) = -1;
        gv(3,1) = +1;gv(3,2) = +2;gv(3,3) = +1;
        
        gh = zeros(3,3);
        gh(1,1) = -1;gh(2,1) = -2;gh(3,1) = -1;
        gh(1,3) = +1;gh(2,3) = +2;gh(3,3) = +1;
    otherwise
        gv = zeros(2,2);
        gv(1,:) = -1;
        gv(2,:) = 1;
        
        gh = zeros(2,2);
        gh(:,1) = -1;
        gh(:,2) = 1;
end

gvf = fft2(gv,R_SIZE,C_SIZE);
ghf = fft2(gh,R_SIZE,C_SIZE);

gvfc = conj(gvf);
ghfc = conj(ghf);

gvf2 = gvfc .* gvf;
ghf2 = ghfc .* ghf;

gf2sum = gvf2 + ghf2;

H_E = double(I_PAN);

for jj = 1 : num_iter
    
    %%% Filter PAN to estimate alpha set
    if jj == 1
        PAN_LP = LPfilter(H_E,ratio);
    else
        PAN_LP = imfilter(H_E,PSF_l,'replicate');
    end
    
    %%% Estimate alpha
    alpha(1,1,:) = estimation_alpha(cat(3,I_MS,ones(size(I_MS,1),size(I_MS,2))),PAN_LP,'global');
    
    It_E = sum(cat(3,I_MS,ones(size(I_MS,1),size(I_MS,2))) .* repmat(alpha,[size(I_MS,1) size(I_MS,2) 1]),3); 

    %%% Edge taper
    H_E = edgetaper(H_E,ones(tap,tap)./((tap)^2));
    It_E = edgetaper(It_E,ones(tap,tap)./((tap)^2));

    %%% Filter Estimation
    PSF = real(fftshift(ifft2(conj(fft2(H_E)).* fft2(It_E)./(abs(fft2(H_E)).^2 + lambda + mu * gf2sum ))));
    
    %%% Thresholding
    PSF(PSF < th) = 0;
    
    %%% Cut using the support dimension and center
    [~, maxIndex] = max(PSF(:));
    [rm, cm] = ind2sub(size(PSF), maxIndex);
    PSF_l = PSF(rm - tap : rm + tap - 1 + sum_tap, cm - tap : cm + tap - 1 + sum_tap);
    PSF_l = PSF_l ./ sum(PSF_l(:));
end

end