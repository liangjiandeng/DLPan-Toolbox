%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
% 
% Interface:
%           ERGAS_index = ERGAS(I1,I2,ratio)
%
% Inputs:
%           I1:             First multispectral image;
%           I2:             Second multispectral image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
% 
% Outputs:
%           ERGAS_index:    ERGAS index.
% References:
%           [Ranchin00]     T. Ranchin and L. Wald, “Fusion of high spatial and spectral resolution images: the ARSIS concept and its implementation,”
%                           Photogrammetric Engineering and Remote Sensing, vol. 66, no. 1, pp. 49–61, January 2000.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ERGAS_index = ERGAS(I1,I2,ratio)

I1 = double(I1);
I2 = double(I2);

Err=I1-I2;
ERGAS_index=0;
for iLR=1:size(Err,3),
    ERGAS_index = ERGAS_index+mean2(Err(:,:,iLR).^2)/(mean2((I1(:,:,iLR))))^2;   
end

ERGAS_index = (100/ratio) * sqrt((1/size(Err,3)) * ERGAS_index);

end