%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Print EPS image.
% 
% Interface:
%           printImage(I_MS,title)
%
% Inputs:
%           I_MS:               Image to print;
%           title:              Filename.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function printImage(I_MS,title)

figure,imshow(I_MS,'Border','tight','InitialMagnification',100);
print(sprintf(title,'.eps'),'-depsc2','-r300');
% print(sprintf(title,'.png'),'-dpng','-r400');

end