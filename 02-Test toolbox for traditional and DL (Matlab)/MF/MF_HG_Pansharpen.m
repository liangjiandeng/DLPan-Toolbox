%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Morphological Pyramid Decomposition using Half-Gradient. 
% 
% Interface:
%           I_Fus_MF_HG = MF_HG_Pansharpen(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_MF_HG:        Morphological Half Gradient (HG) pansharpened image.
% 
% Reference:
%           [Restaino16]        R. Restaino, G. Vivone, M. Dalla Mura, and J. Chanussot, “Fusion of Multispectral and Panchromatic Images Based on Morphological Operators”, 
%                               IEEE Transactions on Image Processing, vol. 25, no. 6, pp. 2882-2895, Jun. 2016.
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
function I_Fus_MF_HG = MF_HG_Pansharpen(I_MS,I_PAN,ratio)

imageLR = double(I_MS);
imageHR = double(I_PAN);

% Equalization
imageHR = repmat(imageHR,[1 1 size(imageLR,3)]);
for ii = 1 : size(imageLR,3)
    imageHR(:,:,ii) = (imageHR(:,:,ii) - mean2(imageHR(:,:,ii))).*(std2(imageLR(:,:,ii))./std2(imageHR(:,:,ii))) + mean2(imageLR(:,:,ii));
end

% Structuring Element  choice
textse= [0 1 0; 1 1 1; 0 1 0];

% Interpolation Method
int_meth='bilinear';

% Number of levels
lev=ceil(log2(ratio))+1;

% Image Construction
P = Pyr_Dec(imageHR,textse,lev,int_meth);

% Fusion   
P_LP = P(:,:,:,lev);
I_Fus_MF_HG = imageLR .* (P(:,:,:,1)./(P_LP+eps));

end