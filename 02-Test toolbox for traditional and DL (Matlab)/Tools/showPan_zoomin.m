%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Visualize and print the panchromatic image.
% 
% Interface:
%           showPan(Pan,print,id,flag_cut_bounds,dim_cut)
%
% Inputs:
%           Pan:                Panchromatic image;
%           print:              Flag. If print == 1, print EPS image;
%           id:                 Identifier (name) of the printed EPS image;
%           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
%           dim_cut:            Define the dimension of the boundary cut;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function IN = showPan_zoomin(Pan,print,id,flag_cut_bounds,dim_cut, location1, location2)

ratio = 4;
if flag_cut_bounds
    %Pan = Pan(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
    Pan = Pan(round(dim_cut/ratio):end-round(dim_cut/ratio),round(dim_cut/ratio):end-round(dim_cut/ratio),:);

end

IN = viewimage(Pan);

if isempty(location2)
    ent=rectangleonimage(IN,location1,1, 3, 3, 3, 1);  % put close-up to up-right corner
    figure,imshow(ent,[])
else
    % type =1 (put to down-left); type =2 (put to down-right); 
    % type =3 (put to up-right); type =4 (put to up-left); 
    ent=rectangleonimage(IN,location1,1, 3, 3, 3, 1);  % put close-up to up-right corner
    ent=rectangleonimage(ent,location2,1, 3, 2, 3, 2);   % put close-up to down-right corner
    figure,imshow(ent,[])
end

if print
    printImage(IN,sprintf('Outputs/%d.eps',id));
end

end