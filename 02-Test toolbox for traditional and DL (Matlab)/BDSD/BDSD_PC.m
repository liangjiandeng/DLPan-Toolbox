%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           BDSD_PC fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Band-Dependent Spatial-Detail (BDSD) model solving an optimization constrained problem. 
% 
% Interface:
%           I_Fus_BDSD = BDSD_PC(I_MS,I_PAN,ratio,S,sensor)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS').
%
% Output:
%           I_Fus_BDSD:     BDSD_PC pansharpened image.
% 
% Reference:
%           [Vivone19]      G. Vivone, “Robust Band-Dependent Spatial-Detail Approaches for Panchromatic Sharpening”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, 2019.
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
function I_Fus_BDSD = BDSD_PC(I_MS,I_PAN,ratio,sensor)

I_MS = double(I_MS);
I_PAN = double(I_PAN);

opts1 = optimset('display','off');

I_GT = imresize(I_MS,1/ratio);%,'nearest');
I_MS_LR = MTF(I_GT,sensor,ratio);
I_PAN_LR = imresize(MTF_PAN(I_PAN,sensor,ratio),1/ratio,'nearest');

I_Fus_BDSD = zeros(size(I_MS));
gamma = zeros(size(I_MS,3)+1,size(I_MS,3));
for ii = 1 : size(I_MS,3)
    h1 = I_GT(:,:,ii);
    h2 = I_MS_LR(:,:,ii);
    H = [I_PAN_LR(:), reshape(I_MS_LR,[size(I_MS_LR,1)*size(I_MS_LR,2), size(I_MS_LR,3)])];
    A = eye(size(I_MS,3)+1);
    A(1,1) = -1;

    gamma(:,ii) = lsqlin(H,h1(:)-h2(:),A,zeros(1,size(I_MS,3)+1),[],[],[],[],[],opts1);
    I_Fus_BDSD(:,:,ii) = I_MS(:,:,ii) + reshape([I_PAN(:),reshape(I_MS,[size(I_MS,1)*size(I_MS,2), size(I_MS,3)])]*gamma(:,ii),[size(I_MS,1) size(I_MS,2)]);
end

end