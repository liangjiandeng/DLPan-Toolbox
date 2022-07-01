%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description:
%                       CSDetails is the Compressive Sensing (CS) approach for Pansharpening proposed in [Vicinanza15].
%
% Interface:
%                       I_Fus_CS = CSDetails(I_MS, I_PAN, I_MS_LR, resize_fact, sensor, TS, ol, n_atoms)
%
% Inputs:
%   I_MS:               Multispectral (MS) original image upsampled to the PAN scale;
%   I_PAN:              Panchromatic (PAN) image;
%   I_MS_LR:            MS original image;
%   ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%   sensor:             String for type of sensor (e.g. 'WV2', 'IKONOS');
%   TS:                 Tiling (dimensions of the patches are TS x TS, e.g. 7 x 7);
%   ol:                 Overlap in pixels between contiguous tile;
%   n_atoms:            max number of representation atoms (default value = 10).
%
% Output:
%   I_Fus_CS:           Fusion image using the CS approach in [Vicinanza15].
%
% References:
%           [Vicinanza15]   M.R. Vicinanza, R. Restaino, G. Vivone, M. Dalla Mura, and J. Chanussot, "A pansharpening method based on the sparse representation of injected details",
%                           IEEE Geoscience and Remote Sensing Letters, vol. 12, no. 1, pp. 180-184, 2015.
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
function I_Fus_CS = CS(I_MS, I_PAN, I_MS_LR, ratio, sensor, TS, ol, n_atoms)

if nargin < 9
    n_atoms = 10;
end

imageLR = double(I_MS);
imageHR = double(I_PAN);
imageLR_LR = double(I_MS_LR);

%%% Equalization
imageHR = repmat (imageHR, [1 1 size(I_MS,3)]);
for ii = 1 : size(imageLR_LR,3)
    %     imageHR(:,:,ii) = equalize_image (imageHR(:,:,ii), imageLR(:,:,ii));
    imageHR(:,:,ii) =  (imageHR(:,:,ii) - mean2(imageHR(:,:,ii))) / std2(imageHR(:,:,ii))...
        * std2(imageLR(:,:,ii)) + mean2(imageLR(:,:,ii));
end

%%% Extract details using MTF-based filters
imageLR_LP = MTF(imageLR, sensor, ratio);
imageLR_D = imageLR - imageLR_LP;
imageHR_LP = MTF(imageHR, sensor, ratio);
for ii = 1:size(imageHR,3)
    imageHR_LP(:,:,ii) = imresize(imresize(imageHR_LP(:,:,ii), 1/ratio, 'nearest'), ratio);
end
imageHR_D = imageHR - imageHR_LP;

%%% Decimation MS
for ii = 1 : size(imageLR,3)
    imageLR_LR(:,:,ii) = double(imresize(imageLR_D(:,:,ii),1/ratio, 'nearest'));
end

%%% Degradation PAN
imageHR_LR = resize_images(imageHR_D, 1, ratio, sensor);

%%% Dictionary learning
[Dh, Dl, ytilde_k] = Dict_Learn(imageHR_D, imageHR_LR, imageLR_LR, ratio, TS, ol);

%%% Sparse coefficient estimation and  HR signal reconstruction
I_Fus_CS = OMP_Rec_Detile(Dl, Dh, ytilde_k, size(imageHR,1), size(imageHR,2), size(imageLR_LR, 3), ratio, ol , TS, n_atoms);


I_Fus_CS = imageLR + I_Fus_CS;

end