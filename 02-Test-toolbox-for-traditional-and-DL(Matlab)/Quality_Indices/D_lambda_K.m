%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Spectral distortion index of the Hybrid Quality with No Reference (HQNR).  
% 
% Interface:
%           Dl = D_lambda_K(fused,ms,ratio,sensor,S)
%
% Inputs:
%           fused:              Pansharpened image;
%           msexp:              MS image resampled to panchromatic scale;
%           sensor:             Type of sensor;
%           ratio:              Resolution ratio;
%           S:                  Block size (optional); Default value: 32.
% 
% Outputs:
%           Dl:                 D_lambda index.
% 
% Reference:
%           [Khan09]            M. M. Khan, L. Alparone, and J. Chanussot, "Pansharpening quality assessment using the modulation transfer functions of instruments,"
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 47, no. 11, pp. 3880-3891, 2009.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dl = D_lambda_K(fused,msexp,ratio,sensor,S)

if (size(fused,1) ~= size(msexp,1) || size(fused,2) ~= size(msexp,2))
    error('The two images must have the same dimensions')
end

[N,M,~] = size(fused);
if (rem(N,S) ~= 0)
    error('number of rows must be multiple of the block size')
end
if (rem(M,S) ~= 0)
    error('number of columns must be multiple of the block size')
end

fused_degraded = MTF(fused,sensor,ratio);

[Q2n_index,~] = q2n(msexp,fused_degraded,S,S);
Dl = 1-Q2n_index;

end

