%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Gaussian Laplacian Pyramid with high pass modulation injection model haze corrected.
% 
% Interface:
%           I_Fus_MTF_GLP_HPM = MTF_GLP_HPM_Haze_min(I_PAN,I_MS,sensor,ratio,decimation)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           sensor:         String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           decimation:     Flag decimation (1: decimated PAN_LP).
%
% Outputs:
%           I_Fus_MTF_GLP_HPM:  Pansharpened image.
% 
% References:
%           [Lolli17]       S. Lolli, L. Alparone, A. Garzelli, and G. Vivone, "Haze correction for contrast-based multispectral pansharpening",
%                           IEEE Geoscience and Remote Sensing Letters, vol. 14, no. 12, pp. 2255-2259, 2017.
%           [Garzelli18]    A. Garzelli, B. Aiazzi, L. Alparone, S. Lolli, and G. Vivone, 
%                           "Multispectral Pansharpening with Radiative Transfer-Based Detail-Injection Modeling for Preserving Changes in Vegetation Cover",
%                           MDPI Remote Sensing, vol. 10, no. 8, pp. 1 - 18, 2018.
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
function I_Fus_MTF_GLP_HPM = MTF_GLP_HPM_Haze_min(I_MS,I_PAN,sensor,ratio,decimation)

if size(I_MS,3) == 4
    prc = 1;
    minMS = zeros(1,1,4);
    B = I_MS(:,:,1);
    G = I_MS(:,:,2);
    R = I_MS(:,:,3);
    NIR = I_MS(:,:,4);
    minMS(1,1,1) = 0.95 * prctile(B(:),prc);
    minMS(1,1,2) = 0.45 * prctile(G(:),prc);
    minMS(1,1,3) = 0.40 * prctile(R(:),prc);
    minMS(1,1,4) = 0.05 * prctile(NIR(:),prc);
else
    minMS = zeros(1,1,size(I_MS,3));
    for ii = 1 : size(I_MS, 3)
       minMS(1,1,ii) = min(min(I_MS(:,:,ii)));  
    end
end

I_PAN_LR = LPfilterGauss(I_PAN,ratio);
w = estimation_alpha(cat(3,ones(size(I_PAN_LR)),I_MS),I_PAN_LR,'global');
wp = w' * [1;squeeze(minMS)]; 

L = repmat(minMS, [size(I_MS,1) size(I_MS,2)]);
Lp = wp .* ones([size(I_MS,1) size(I_MS,2)]);

imageHR = double(I_PAN);
I_MS = double(I_MS);

%%% Equalization
imageHR = repmat(imageHR,[1 1 size(I_MS,3)]);

PAN_LP = MTF(imageHR,sensor,ratio);

if decimation
    for ii = 1 : size(I_MS,3)
        t = imresize(PAN_LP(:,:,ii),1/ratio,'nearest');
        PAN_LP(:,:,ii) = interp23tap(t,ratio);
    end
end

P_PL = (imageHR - Lp) ./ (PAN_LP - Lp + eps);

MS_L = I_MS - L;

I_Fus_MTF_GLP_HPM = MS_L .* P_PL + L;

end