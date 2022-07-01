%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF_GLP_HPM fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Modulation Transfer Function - Generalized Laplacian Pyramid (MTF-GLP) with High Pass Modulation (HPM) injection model algorithm. 
% 
% Interface:
%           I_Fus_MTF_GLP_HPM = MTF_GLP_HPM(I_MS,I_PAN,sensor,ratio)
%
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_MTF_GLP_HPM:  MTF_GLP_HPM pansharpened image.
% 
% References:
%           [Aiazzi03]          B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “An MTF-based spectral distortion minimizing model for Pan-sharpening
%                               of very high resolution multispectral images of urban areas,” in Proceedings of URBAN 2003: 2nd GRSS/ISPRS Joint Workshop on
%                               Remote Sensing and Data Fusion over Urban Areas, 2003, pp. 90–94.
%           [Aiazzi06]          B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,”
%                               Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
%           [Vivone14a]         G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral
%                               image pansharpening,” IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
%           [Vivone15]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%           [Alparone17]        L. Alparone, A. Garzelli, and G. Vivone, "Intersensor statistical matching for pansharpening: Theoretical issues and practical solutions",
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 8, pp. 4682-4695, 2017.
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
function I_Fus_MTF_GLP_HPM = MTF_GLP_HPM(I_MS,I_PAN,sensor,ratio)

imageHR = double(I_PAN);
I_MS = double(I_MS);

%%% Equalization
imageHR = repmat(imageHR,[1 1 size(I_MS,3)]);

for ii = 1 : size(I_MS,3)    
  imageHR(:,:,ii) = (imageHR(:,:,ii) - mean2(imageHR(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(LPfilterGauss(imageHR(:,:,ii),ratio))) + mean2(I_MS(:,:,ii));  
end


h = genMTF(ratio, sensor, size(I_MS,3));

PAN_LP = zeros(size(I_MS));
for ii = 1 : size(I_MS,3)
    PAN_LP(:,:,ii) = imfilter(imageHR(:,:,ii),real(h(:,:,ii)),'replicate');
    t = imresize(PAN_LP(:,:,ii),1/ratio,'nearest');
    PAN_LP(:,:,ii) = interp23tap(t,ratio);
end

I_Fus_MTF_GLP_HPM = I_MS .* (imageHR ./ (PAN_LP + eps));

end