%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GS2_GLP fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Gram-Schmidt (GS) mode 2 algorithm with Generalized Laplacian Pyramid (GLP) decomposition.
% 
% Interface:
%           I_Fus_GS2_GLP = GS2_GLP(I_MS,I_PAN,ratio,sensor)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:         String for type of sensor (e.g. 'WV2','IKONOS').
%
% Outputs:
%           I_Fus_GS2_GLP:  GS2_GLP pasharpened image.
% 
% References:
%           [Aiazzi06]      B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,”
%                           Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
%           [Alparone07]    L. Alparone, L. Wald, J. Chanussot, C. Thomas, P. Gamba, and L. M. Bruce, “Comparison of pansharpening algorithms: Outcome
%                           of the 2006 GRS-S Data Fusion Contest,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3012–3021,
%                           October 2007.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
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
function I_Fus_GS2_GLP = GS2_GLP(I_MS,I_PAN,ratio,sensor)

imageLR = double(I_MS);
imageHR = double(I_PAN);

imageHR = repmat(imageHR,[1 1 size(imageLR,3)]);

h = genMTF(ratio, sensor, size(I_MS,3));

PAN_LP = zeros(size(I_MS));
for ii = 1 : size(I_MS,3)
    PAN_LP(:,:,ii) = imfilter(imageHR(:,:,ii),real(h(:,:,ii)),'replicate');
    t = imresize(PAN_LP(:,:,ii),1/ratio,'nearest');
    PAN_LP(:,:,ii) = interp23tap(t,ratio);
end

PAN_LP = double(PAN_LP);

%%% Coefficients
g = ones(1,size(I_MS,3));
for ii = 1 : size(I_MS,3)
    h = imageLR(:,:,ii);
    h2 = PAN_LP(:,:,ii);
    c = cov(h2(:),h(:));
    g(ii) = c(1,2)/var(h2(:));
end

%%% Detail Extraction
delta = imageHR - PAN_LP;

I_Fus_GS2_GLP = zeros(size(imageLR));

for ii = 1 : size(imageLR,3)
    I_Fus_GS2_GLP(:,:,ii) = imageLR(:,:,ii) + delta(:,:,ii) .* g(ii);
end

end