%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           spatial Correlation Coefficient (sCC).
% 
% Interface:
%           [sCC,SCCMap] = SCC(I_F,I_GT)
%
% Inputs:
%           I_F:        Fused image;
%           I_GT:       Ground-truth image.
% 
% Outputs:
%           sCC:        spatial correlation coefficient;
%           SCCMap:     Image of sCC values.
% 
% Reference:
%           [Vivone15]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                       IEEE Transaction on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565-2586, May 2015.
%           [Vivone20]  G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                       IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sCC,SCCMap]=SCC(I_F,I_GT)

Im_Lap_F = zeros(size(I_F,1)-2,size(I_F,2)-2,size(I_F,3));
for idim=1:size(I_F,3)
    Im_Lap_F_y= imfilter(I_F(2:end-1,2:end-1,idim),fspecial('sobel'));
    Im_Lap_F_x= imfilter(I_F(2:end-1,2:end-1,idim),fspecial('sobel')');
    Im_Lap_F(:,:,idim) = sqrt(Im_Lap_F_y.^2+Im_Lap_F_x.^2);
end

Im_Lap_GT = zeros(size(I_GT,1)-2,size(I_GT,2)-2,size(I_GT,3));
for idim=1:size(I_GT,3)
    Im_Lap_GT_y= imfilter(I_GT(2:end-1,2:end-1,idim),fspecial('sobel'));
    Im_Lap_GT_x= imfilter(I_GT(2:end-1,2:end-1,idim),fspecial('sobel')');
    Im_Lap_GT(:,:,idim) = sqrt(Im_Lap_GT_y.^2+Im_Lap_GT_x.^2);
end

sCC=sum(sum(sum(Im_Lap_F.*Im_Lap_GT)));
sCC = sCC/sqrt(sum(Im_Lap_F(:).^2));
sCC = sCC/sqrt(sum(Im_Lap_GT(:).^2));

SCCMap=sum(Im_Lap_F.*Im_Lap_GT,3)/sqrt(sum(Im_Lap_GT(:).^2))...
    /sqrt(sum(Im_Lap_GT(:).^2));

end