%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Brovey data fusion with haze correction
% 
% Interface:
%           I_Fus_Brovey_Reg = BroveyRegHazeMin(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_Brovey_Reg:  Pansharpened image.
% 
% References:
%           [Lolli17]       S. Lolli, L. Alparone, A. Garzelli, and G. Vivone, "Haze correction for contrast-based multispectral pansharpening",
%                           IEEE Geoscience and Remote Sensing Letters, vol. 14, no. 12, pp. 2255-2259, 2017.
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
function I_Fus_Brovey_Reg = BroveyRegHazeMin(I_MS,I_PAN,ratio)

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

L = repmat(minMS, [size(I_MS,1) size(I_MS,2)]);

imageLR = double(I_MS);
imageHR = double(I_PAN);

imageHR_LR = LPfilterGauss(imageHR,ratio);

h = estimation_alpha(imageLR,imageHR_LR,'global');

alpha(1,1,:) = h;

I = sum((imageLR - L) .* repmat(alpha,[size(I_MS,1) size(I_MS,2) 1]),3); 

imageHR = (imageHR - mean2(imageHR_LR)).*(std2(I)./std2(imageHR_LR)) + mean2(I);  

I_MS_L = imageLR - L;
I_MS_L(I_MS_L < 0) = 0;

I_Fus_Brovey_Reg = I_MS_L .* repmat(imageHR./(I+eps),[1 1 size(imageLR,3)]) + L;

end