%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GS_Segm fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the segmentation-based version of the Gram-Schmidt algorithm.
% 
% Interface:
%           PanSharpenedImage = GS_Segm(I_MS,I_PAN,I_LR_input,S)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale
%           I_PAN:      PAN image
%           I_LR_input: Low Resolution PAN Image 
%           S:          Segmentation
%
% Outputs:
%           PanSharpenedImage:  Pasharpened image
% 
% Reference:
%           [Restaino17]    R. Restaino, M. Dalla Mura, G. Vivone, J. Chanussot, “Context-Adaptive Pansharpening Based on Image Segmentation”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 2, pp. 753–766, February 2017.
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
function PanSharpenedImage = GS_Segm(I_MS,I_PAN,I_LR_input,S)
I_MS = double(I_MS);

I_PAN = repmat(double(I_PAN), [1, 1, size(I_MS,3)]);
I_LR_input = double(I_LR_input);
if size(I_LR_input, 3) == 1
    I_LR_input = repmat(I_LR_input, [1, 1, size(I_MS,3)]);
end
if size(I_LR_input, 3) ~= size(I_PAN, 3)
    error('I_LP should have the same number of bands as PAN');
end

DetailsHRPan = I_PAN - I_LR_input;

Coeff = zeros(size(I_MS));
labels = unique(S);

for ii = 1: size(I_MS,3)
    MS_Band = squeeze(I_MS(:,:,ii));
    I_LR_Band = squeeze(I_LR_input(:,:,ii));
    Coeff_Band = zeros(size(I_LR_Band));
    for il=1:length(labels)
        idx = S==labels(il);
        c = cov(I_LR_Band(idx),MS_Band(idx));
        Coeff_Band(idx) = c(1,2)/var(I_LR_Band(idx));
    end
    Coeff(:,:,ii) = Coeff_Band;
end

PanSharpenedImage = Coeff .* DetailsHRPan + I_MS;

end