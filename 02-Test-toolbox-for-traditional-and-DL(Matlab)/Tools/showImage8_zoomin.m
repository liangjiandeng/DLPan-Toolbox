%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Visualize and print an eight-band multispectral image.
% 
% Interface:
%           showImage8(I_MS,print,id,flag_cut_bounds,dim_cut,th_values,L)
%
% Inputs:
%           I_MS:               Eight band multispectral image;
%           print:              Flag. If print == 1, print EPS image;
%           id:                 Identifier (name) of the printed EPS image;
%           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
%           dim_cut:            Define the dimension of the boundary cut;
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
%           L:                  Radiomatric resolution of the input image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function showImage8_zoomin(I_MS,print,id,flag_cut_bounds,dim_cut,th_values,L, location1, location2)

if flag_cut_bounds
    I_MS = I_MS(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
end

if th_values
    I_MS(I_MS > 2^L) = 2^L;
    I_MS(I_MS < 0) = 0;
end

if id == 1
    IMN = viewimage(I_MS(:,:,[1,3,5]));
    IMN = IMN(:,:,3:-1:1);
else
    IMN = viewimage(I_MS(:,:,[1,3,5]),[0.01 0.995]);
    IMN = IMN(:,:,3:-1:1);
end

if isempty(location2)
    ent=rectangleonimage(IMN,location1,1, 3, 3, 3, 1);  % put close-up to up-right corner
    figure,imshow(ent,[])
else
    % type =1 (put to down-left); type =2 (put to down-right); 
    % type =3 (put to up-right); type =4 (put to up-left); 
    ent=rectangleonimage(IMN,location1,1, 3, 3, 3, 1);  % put close-up to up-right corner
    ent=rectangleonimage(ent,location2,1, 3, 2, 3, 2);   % put close-up to down-right corner
    figure,imshow(ent,[])
end

if print
    printImage(IMN,sprintf('Outputs/%d.eps',id));
end

end