%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           interp23tap interpolates the image I_Interpolated using a polynomial with 23 coefficients interpolator. 
% 
% Interface:
%           I_Interpolated = interp23tap(I_Interpolated,ratio)
%
% Inputs:
%           I_Interpolated: Image to interpolate;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Resize factors power of 2.
%
% Outputs:
%           I_Interpolated: Interpolated image.
% 
% References:
%           [Aiazzi02]      B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on
%                           oversampled multiresolution analysis,? IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300?2312, October
%                           2002.
%           [Aiazzi13]      B. Aiazzi, S. Baronti, M. Selva, and L. Alparone, “Bi-cubic interpolation for shift-free pan-sharpening,? ISPRS Journal of Photogrammetry
%                           and Remote Sensing, vol. 86, no. 6, pp. 65?76, December 2013.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms?, 
%                           IEEE Transaction on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565?2586, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Interpolated = interp23tap(I_Interpolated,ratio)

if (2^round(log2(double(ratio))) ~= ratio)
    disp('Error: Only resize factors power of 2');
    return;
end 

[r,c,b] = size(I_Interpolated);

CDF23 = 2.*[0.5 0.305334091185 0 -0.072698593239 0 0.021809577942 0 -0.005192756653 0 0.000807762146 0 -0.000060081482];
CDF23 = [fliplr(CDF23(2:end)) CDF23];
BaseCoeff = CDF23;
first = 1;

for z = 1 : ratio/2

    I1LRU = zeros((2^z) * r, (2^z) * c, b);
    
    if first
        I1LRU(2:2:end,2:2:end,:) = I_Interpolated;
        first = 0;
    else
        I1LRU(1:2:end,1:2:end,:) = I_Interpolated;
    end

    for ii = 1 : b
        t = I1LRU(:,:,ii); 
        t = imfilter(t',BaseCoeff,'circular'); 
        I1LRU(:,:,ii) = imfilter(t',BaseCoeff,'circular'); 
    end
    
    I_Interpolated = I1LRU;
    
end

end

