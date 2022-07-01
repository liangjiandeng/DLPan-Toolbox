%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Visualize and print the original four-band multispectral image.
% 
% Interface:
%           showImage4LR(I_F,print,id,flag_cut_bounds,dim_cut,thvalues,L,ratio)
%
% Inputs:
%           I_MS:               Four band multispectral image;
%           print:              Flag. If print == 1, print EPS image;
%           id:                 Identifier (name) of the printed EPS image;
%           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
%           dim_cut:            Define the dimension of the boundary cut;
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
%           L:                  Radiomatric resolution of the input image;
%           ratio:              Resize factor.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function showImage4LR(I_MS,print,id,flag_cut_bounds,dim_cut,th_values,L,ratio)

if flag_cut_bounds
    I_MS = I_MS(round(dim_cut/ratio):end-round(dim_cut/ratio),round(dim_cut/ratio):end-round(dim_cut/ratio),:);
end

if th_values
    I_MS(I_MS > 2^L) = 2^L;
    I_MS(I_MS < 0) = 0;
end

IMN = viewimage(I_MS(:,:,1:3));
IMN = IMN(:,:,3:-1:1);

if print
    printImage(IMN,sprintf('Outputs/%d.eps',id));
end

end