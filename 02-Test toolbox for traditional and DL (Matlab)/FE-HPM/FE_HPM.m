%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           FE_HPM fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the high pass modulation injection model and the estimated filter via deconvolution. 
% 
% Interface:
%           [I_Fus,D,PSF_l] = FE_HPM(I_MS,I_PAN,ratio,tap,lambda,mu,th,num_iter,filtername)
%
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%           tap:                Filter support;
%           lambda:             Coefficient for weighting the energy regularization term;
%           mu:                 Coefficient for weighting the derivative regularization terms;
%           th:                 Threshold on the kernel (it cuts to 0 values below threshold);
%           num_iter_max:       Max number of iteration (at least 3; not sensitive);    
%           filtername:         Kind of derivative (default: 'Basic')       
%
% Outputs:
%           I_Fus,D:            Pansharpened image;
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
function [I_Fus,PSF_l] = FE_HPM(I_MS,I_PAN,ratio,tap,lambda,mu,th,num_iter,filtername)

imageHR = double(I_PAN);
I_MS = double(I_MS);
nBands = size(I_MS,3);

%%% Equalization
imageHR = repmat(imageHR,[1 1 size(I_MS,3)]);
for ii = 1 : size(I_MS,3)    
  imageHR(:,:,ii) = (imageHR(:,:,ii) - mean2(imageHR(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(imageHR(:,:,ii))) + mean2(I_MS(:,:,ii));  
end

PSF_l = FE(I_MS,I_PAN,ratio,tap,lambda,mu,th,num_iter,filtername);

PAN_LP = zeros(size(imageHR));
for ii = 1 : nBands
    PAN_LP(:,:,ii) = imfilter(imageHR(:,:,ii),PSF_l,'replicate');
    t = imresize(PAN_LP(:,:,ii),1/ratio,'nearest');
    PAN_LP(:,:,ii) = interp23tap(t,ratio);
end

PAN_LP = double(PAN_LP);

I_Fus = I_MS .* (imageHR ./ (PAN_LP + eps));

end