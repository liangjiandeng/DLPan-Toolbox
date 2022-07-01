%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF filters the image I_MS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the MultiSpectral (MS) sensor. 
% 
% Interface:
%           I_Filtered = MTF(I_MS,sensor,ratio)
%
% Inputs:
%           I_MS:           MS image;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
%           ratio:          Scale ratio between MS and PAN.
%
% Outputs:
%           I_Filtered:     Output filtered MS image.
% 
% References:
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
function I_Filtered = MTF(I_MS,sensor,ratio)

h = genMTF(ratio, sensor, size(I_MS,3));

I_MS_LP = zeros(size(I_MS));
for ii = 1 : size(I_MS,3)
    I_MS_LP(:,:,ii) = imfilter(I_MS(:,:,ii),real(h(:,:,ii)),'replicate');
end

I_Filtered = double(I_MS_LP);

end