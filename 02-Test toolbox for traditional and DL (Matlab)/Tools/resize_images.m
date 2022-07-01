%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description:
%            Resize_images generates the low resolution panchromatic (PAN) and multispectral (MS) images according to Wald's protocol. 
% 
% Interface:
%           [I_MS_LR, I_PAN_LR] = resize_images(I_MS,I_PAN,ratio,sensor)
% 
% Inputs:
%       	I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:             String for type of sensor (e.g. 'WV2', 'IKONOS').
% 
% Outputs:
%           I_MS_LR:            Low Resolution MS image;
%           I_PAN_LR:           Low Resolution PAN image.
% 
% References:
%           [Wald97]            L. Wald, T. Ranchin, and M. Mangolini, “Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images,”
%                               Photogrammetric Engineering and Remote Sensing, vol. 63, no. 6, pp. 691–699, June 1997.
%           [Aiazzi02]          B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on
%                               oversampled multiresolution analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October
%                               2002.
%           [Aiazzi06]          B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,”
%                               Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
%           [Vivone14a]         G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral
%                               image pansharpening,” IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
%           [Vivone15]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I_MS_LR, I_PAN_LR] = resize_images(I_MS,I_PAN,ratio,sensor)

I_MS = double(I_MS);
I_PAN = double(I_PAN);
  
I_MS_LP = MTF(I_MS,sensor,ratio);

%%% Decimation MS
I_MS_LP_D = zeros(round(size(I_MS,1)/ratio),round(size(I_MS,2)/ratio),size(I_MS,3));
for idim = 1 : size(I_MS,3)
    I_MS_LP_D(:,:,idim) = imresize(I_MS_LP(:,:,idim),1/ratio,'nearest');
end

I_MS_LR = double(I_MS_LP_D);

I_PAN_LR = imresize(I_PAN, 1/ratio);

end