%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Morphological Pyramid Decomposition using Half-Gradient. 
% 
% Interface:
%           P = Pyr_Dec(Im,textse,lev,int_meth)
%
% Inputs:
%           Im:                 Image to decompose;
%           textse:             Structuring Element;
%           lev:                Number of decomposition levels;
%           int_meth:           Interpolation method.
%
% Outputs:
%           P:                  Morphological Pyramid using Half-Gradient.
% 
% References:
%           [Vivone14]          G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral
%                               image pansharpening,” IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
%           [Vivone15]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565-2586, May 2015.
%           [Restaino16]        R. Restaino, G. Vivone, M. Dalla Mura, and J. Chanussot, “Fusion of Multispectral and Panchromatic Images Based on Morphological Operators”, 
%                               IEEE Transactions on Image Processing, vol. 25, no. 6, pp. 2882-2895, Jun. 2016.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  P = Pyr_Dec(Im,textse,lev,int_meth)

P(:,:,:,1) = Im;
Sizes(1,:)=[size(Im,1), size(Im,2)];
imageI_new=P(:,:,:,1);
first=1;

for ii = 2 : lev
    
    imageI_old = imageI_new;
    clear imageI_new

    %  Half Gradient
    PD = imdilate(imageI_old,textse);
    PE= imerode(imageI_old,textse);
    rho_minus=imageI_old-PE;
    rho_plus=PD-imageI_old;
    D=rho_minus-rho_plus;
    PS = imageI_old -0.5*D;
    % PS = 0.5*squeeze(PD+PE); %equivalently
            
    % Downsampling
    if first
        for il=1:size(imageI_old,3)
            imageI_new(:,:,il)=PS(2:2:end,2:2:end,il);
        end
        first=0;
    else
        for il=1:size(imageI_old,3)
            imageI_new(:,:,il)=PS(1:2:end,1:2:end,il);
        end
    end
    Sizes(ii,:)=[size(imageI_new,1) size(imageI_new,1)];
    imageI_resized_old=imageI_new;
    for ir=ii:-1:2,
        for il=1:size(Im,3)
            imageI_resized_new(:,:,il)  = imresize(imageI_resized_old(:,:,il),[Sizes(ir-1,1) Sizes(ir-1,2)],int_meth);
        end
        imageI_resized_old=imageI_resized_new;
        clear imageI_resized_new
    end
    
    if sum(isfinite(imageI_resized_old(:)))~=numel(imageI_resized_old)
        P(:,:,:,1:lev) =repmat(P(:,:,:,1),1,1,1,lev);
        break
    else
        P(:,:,:,ii) = imageI_resized_old;
    end
    
    clear imageI_resized_old
end

end

