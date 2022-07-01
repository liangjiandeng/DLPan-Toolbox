%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Estimation coefficients linear regression model. 
% 
% Interface:
%           alpha = estimation_alpha(I_MS,I_PAN,type_estimation)
% 
% Inputs:
%           I_MS:               MS image upsampled at PAN scale;
%           I_PAN:              PAN image;
%           type_estimation:    Type of estimation (i.e. local or global).
%
% Outputs:
%           alpha:              Coefficients estimated by the linear regression model.
% 
% References:
%           [Vivone14]          G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral
%                               image pansharpening,” IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function alpha = estimation_alpha(I_MS,I_PAN,type_estimation)

if strcmp(type_estimation,'global')
    %%%%%%%% Global estimation
    IHc = reshape(I_PAN,[numel(I_PAN) 1]);
    ILRc = reshape(I_MS,[size(I_MS,1)*size(I_MS,2) size(I_MS,3)]);
    alpha = ILRc\IHc;
else
    %%%%%%%% Local estimation
    block_win = 32;
    alphas = zeros(size(I_MS,3),1);
    cont_bl = 0;
    for ii = 1 : block_win : size(I_MS,1)
        for jj = 1 : block_win : size(I_MS,2)
                imHRbl = I_PAN(ii : min(size(I_MS,1),ii + block_win - 1), jj : min(size(I_MS,2),jj + block_win - 1));
                imageLRbl = I_MS(ii : min(size(I_MS,1),ii + block_win - 1), jj : min(size(I_MS,2),jj + block_win - 1),:);
                imageHRc = reshape(imHRbl,[numel(imHRbl) 1]);
                ILRc = reshape(imageLRbl,[size(imageLRbl,1).*size(imageLRbl,2) size(imageLRbl,3)]);
                alphah = ILRc\imageHRc;
                alphas = alphas + alphah;
                cont_bl = cont_bl + 1;
        end
    end
    alpha = alphas/cont_bl;
end

end