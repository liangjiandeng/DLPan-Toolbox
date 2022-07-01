%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Visualize all the images applying the same stretching for visual comparison.
% 
% Interface:
%           MatrixPrint = showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,flagPAN)
%
% Inputs:
%           MatrixImage:        Matrix that contains all the images to visualize; Size: [M x N x B x Z], where [M x N] is the
%                               dimension of a single image band, B represents the number of bands for each image, and Z is the number of images to plot.
%           titleImages:        Vector of strings that represents the titles for each image to plot; Size: [1 x Z].
%           vect_index_RGB:     Identify the bands to plot to obtain an RGB representation of the multispectral data;
%           flag_cut_bounds:    Cut the boundaries of the images to plot;
%           dim_cut:            Define the dimension of the boundary cut;
%           flagPAN:            Flag. If flagPAN == 1, the first image to plot is the panchromatic image otherwise it is the ground-truth.
%
% Outputs:
%           MatrixPrint:        Matrix, with the same structure of MatrixImage, which contains the plotted images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MatrixPrint = showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,flagPAN)

if flag_cut_bounds
    MatrixImageCat = zeros(numel(dim_cut:size(MatrixImage,1)-dim_cut),numel(dim_cut:size(MatrixImage,2)-dim_cut),size(MatrixImage,3),size(MatrixImage,4));
    for ii = 1 : size(MatrixImageCat,4)
        t = MatrixImage(:,:,:,ii);
        MatrixImageCat(:,:,:,ii) = t(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
    end
else
    MatrixImageCat = MatrixImage;
end

[r,c,~] = size(MatrixImageCat(:,:,:,1));

if flagPAN
    T = [];
    for ii = 2 : size(MatrixImageCat,4)
        T = cat(2,T,MatrixImageCat(:,:,vect_index_RGB,ii));
    end    
else
    T = [];
    for ii = 1 : size(MatrixImageCat,4)
        T = cat(2,T,MatrixImageCat(:,:,vect_index_RGB,ii));
    end
end

IMN = viewimage2(T);

if flagPAN
    MatrixPrint = zeros(size(MatrixImageCat(:,:,vect_index_RGB,:)));
    MatrixPrint(:,:,:,1) = viewimage2(MatrixImageCat(:,:,vect_index_RGB,1));
    ind_c = 1;
    for ii = 2 : size(MatrixImageCat,4)   
        MatrixPrint(:,:,:,ii) = IMN(1 : r,ind_c : ind_c + c - 1,:);
        ind_c = ind_c + c;
    end    
else
    MatrixPrint = zeros(size(MatrixImageCat(:,:,vect_index_RGB,:)));
    ind_c = 1;
    for ii = 1 : size(MatrixImageCat,4)   
        MatrixPrint(:,:,:,ii) = IMN(1 : r,ind_c : ind_c + c - 1,:);
        ind_c = ind_c + c;
    end
end

ha = tight_subplot(5,5,[.06 .03],[.01 .06],[.01 .01]);
% ha = tight_subplot(5,5,[.02 0],[.01 .03],[.0 .0]);

for ii = 1 : size(MatrixImageCat,4)
    axes(ha(ii)); imshow(MatrixPrint(:,:,:,ii),[]);
    title(ha(ii),titleImages{ii});
end
   
end