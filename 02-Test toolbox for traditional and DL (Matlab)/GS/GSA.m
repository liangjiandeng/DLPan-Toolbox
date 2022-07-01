%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GSA fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Gram-Schmidt Adaptive (GSA) algorithm.
% 
% Interface:
%           I_Fus_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image;
%           I_MS_LR:    MS image;
%           ratio:      Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_GSA:  GSA pasharpened image.
% 
% References:
%           [Aiazzi07]  B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan
%                       data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, October 2007.
%           [Vivone15]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                       IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%           [Vivone20]  G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                       IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2019
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio)

imageLR = double(I_MS);
imageHR = double(I_PAN);
imageLR_LP = double(I_MS_LR);

%%% Remove means from imageLR
imageLR0 = zeros(size(I_MS));
for ii = 1 : size(I_MS,3), imageLR0(:,:,ii) = imageLR(:,:,ii) - mean2(imageLR(:,:,ii)); end

%%% Remove means from imageLR_LP
imageLR_LP0 = zeros(size(I_MS_LR));
for ii = 1 : size(I_MS_LR,3), imageLR_LP0(:,:,ii) = imageLR_LP(:,:,ii) - mean2(imageLR_LP(:,:,ii)); end


%%% Intensity
imageHR0 = imageHR - mean2(imageHR);
imageHR0 = LPfilterPlusDec(imageHR0,ratio);
alpha(1,1,:) = estimation_alpha(cat(3,imageLR_LP0,ones(size(I_MS_LR,1),size(I_MS_LR,2))),imageHR0,'global');
I = sum(cat(3,imageLR0,ones(size(I_MS,1),size(I_MS,2))) .* repmat(alpha,[size(I_MS,1) size(I_MS,2) 1]),3); 

%%% Remove mean from I
I0 = I - mean2(I);

%%% Coefficients
g = ones(1,1,size(I_MS,3)+1);
for ii = 1 : size(I_MS,3)
    h = imageLR0(:,:,ii);
    c = cov(I0(:),h(:));
    g(1,1,ii+1) = c(1,2)/var(I0(:));
end

imageHR = imageHR - mean2(imageHR);

%%% Detail Extraction
delta = imageHR - I0;
deltam = repmat(delta(:),[1 size(I_MS,3)+1]);

%%% Fusion
V = I0(:);
for ii = 1 : size(I_MS,3)
    h = imageLR0(:,:,ii);
    V = cat(2,V,h(:));
end

gm = zeros(size(V));
for ii = 1 : size(g,3)
    gm(:,ii) = squeeze(g(1,1,ii)) .* ones(size(I_MS,1).*size(I_MS,2),1);
end

V_hat = V + deltam .* gm;

%%% Reshape fusion result
I_Fus_GSA = reshape(V_hat(:,2:end),[size(I_MS,1) size(I_MS,2) size(I_MS,3)]);

%%% Final Mean Equalization
for ii = 1 : size(I_MS,3)
    h = I_Fus_GSA(:,:,ii);
    I_Fus_GSA(:,:,ii) = h - mean2(h) + mean2(imageLR(:,:,ii));
end

end