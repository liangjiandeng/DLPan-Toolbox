%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF_GLP_FS fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Modulation Transfer Function - Generalized Laplacian Pyramid (MTF-GLP) and a new Full Resolution Regression-based injection model. 
% 
% Interface:
%           I_Fus_MTF_GLP_FS = MTF_GLP_FS(I_MS,I_PAN,sensor,ratio)
%
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_MTF_GLP_FS:	Pansharpened image.
% 
% Reference:
%           [Vivone18]          G. Vivone, R. Restaino,and J. Chanussot, "Full scale regression-based injection coefficients for panchromatic sharpening," 
%                               IEEE Transactions on Image Processing, vol. 27, no. 7, pp. 3418-3431, Jul. 2018.
%           [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                               IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2019
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_MTF_GLP_FS = MTF_GLP_FS(I_MS,I_PAN,sensor,ratio)

imageHR = double(I_PAN);
I_MS = double(I_MS);

h = genMTF(ratio, sensor, size(I_MS,3));

I_Fus_MTF_GLP_FS = zeros(size(I_MS));
for ii = 1 : size(I_MS,3)
    %%% Low resolution PAN image
    PAN_LP = imfilter(imageHR,real(h(:,:,ii)),'replicate');
    t = imresize(PAN_LP,1/ratio,'nearest');    
    PAN_LP = interp23tap(t,ratio);
    
    %%% Injection coefficient for band ii    
    MSB = I_MS(:,:,ii);
    CMSPAN = cov(MSB(:), imageHR(:));    
    CPANPANLR = cov(PAN_LP(:), imageHR(:));
    gFS = CMSPAN(1,2)./CPANPANLR(1,2);
    
    %%% Fusion rule
    I_Fus_MTF_GLP_FS(:,:,ii) = I_MS(:,:,ii) + gFS .* (imageHR - PAN_LP);
end

end