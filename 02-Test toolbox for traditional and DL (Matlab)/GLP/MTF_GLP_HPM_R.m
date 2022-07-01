%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           A Regression-Based High-Pass Modulation Pansharpening Approach (Global Version) 
% 
% Interface:
%           I_Fus_MTF_GLP_HPM_R = MTF_GLP_HPM_R(I_MS,I_PAN,sensor,ratio)
%
% Inputs:
%           I_MS:                   MS image upsampled at PAN scale;
%           I_PAN:                  PAN image;
%           sensor:                 String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:                  Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_MTF_GLP_HPM_R:    Pansharpened image.
% 
% Reference:
%           [Vivone18]              G. Vivone, R. Restaino, and J. Chanussot, "A regression-based high-pass modulation pansharpening approach," 
%                                   IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 2, pp. 984-996, Feb. 2018.
%           [Vivone20]              G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                                   IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2019
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_MTF_GLP_HPM_R = MTF_GLP_HPM_R(I_MS,I_PAN,sensor,ratio)

imageHR = double(I_PAN);
I_MS = double(I_MS);

h = genMTF(ratio, sensor, size(I_MS,3));

I_Fus_MTF_GLP_HPM_R = zeros(size(I_MS));
for ii = 1 : size(I_MS,3)
    %%% Low resolution PAN image
    PAN_LP = imfilter(imageHR,real(h(:,:,ii)),'replicate');
    t = imresize(PAN_LP,1/ratio,'nearest');
    PAN_LP = interp23tap(t,ratio);
    
    %%%% Regression coefficients
    MSB = I_MS(:,:,ii);
	C = cov(MSB(:),PAN_LP(:));
    g = C(1,2)./C(2,2);
    cb = mean(MSB(:))./g - mean(imageHR(:));
        
    %%% Fusion rule
    I_Fus_MTF_GLP_HPM_R(:,:,ii) = I_MS(:,:,ii) .* (imageHR + cb) ./ (PAN_LP + cb + eps);
end

end