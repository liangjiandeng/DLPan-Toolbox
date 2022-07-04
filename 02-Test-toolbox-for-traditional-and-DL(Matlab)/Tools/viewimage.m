%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Visualization [3-2-1] of images with 3 bands by exploiting linear stretching and fixing the saturation. 
% 
% Interface:
%           ImageToView = viewimage(ImageToView,tol)
%
% Inputs:
%           ImageToView:    Image to view;
%           tol:            Saturation; Default values: [0.01 0.99] equal for all the three bands.
%
% Outputs:
%           ImageToView:    Image to view.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ImageToView = viewimage(ImageToView,tol1,tol2,tol3)

iptsetpref('ImshowBorder', 'tight')
ImageToView = double(ImageToView);
L=size(ImageToView,3);
if (L<3)
    ImageToView=ImageToView(:,:,[1 1 1]);
end

if nargin == 1
    tol1 = [0.01 0.99];
end
if nargin <= 2
    tol = [tol1;tol1;tol1];
    ImageToView = linstretch(ImageToView,tol);
    figure,imshow(ImageToView(:,:,3:-1:1),[])
elseif nargin == 4
    if sum(tol1(2)+tol2(2)+tol3(2)) <= 3
        tol = [tol1;tol2;tol3];
        ImageToView = linstretch(ImageToView,tol);
        figure,imshow(ImageToView(:,:,3:-1:1),[])
    else
        tol = [tol1;tol2;tol3];
        [N,M,~] = size(ImageToView);
        NM = N*M;
        for i=1:3
            b = reshape(double(uint16(ImageToView(:,:,i))),NM,1);
            b(b<tol(i,1))=tol(i,1);
            b(b>tol(i,2))=tol(i,2);
            b = (b-tol(i,1))/(tol(i,2)-tol(i,1));
            ImageToView(:,:,i) = reshape(b,N,M);
        end
        figure,imshow(ImageToView(:,:,3:-1:1),[])
    end
end

iptsetpref('ImshowBorder', 'loose')

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Linear Stretching. 
% 
% Interface:
%           ImageToView = linstretch(ImageToView,tol)
%
% Inputs:
%           ImageToView:    Image to stretch;
%           tol:            ;
%
% Outputs:
%           ImageToView:    Stretched image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ImageToView = linstretch(ImageToView,tol)

[N,M,~] = size(ImageToView);
NM = N*M;
for i=1:3
    b = reshape(double(uint16(ImageToView(:,:,i))),NM,1);
    [hb,levelb] = hist(b,max(b)-min(b));
    chb = cumsum(hb);
    t(1)=ceil(levelb(find(chb>NM*tol(i,1), 1 )));
    t(2)=ceil(levelb(find(chb<NM*tol(i,2), 1, 'last' )));
    b(b<t(1))=t(1);
    b(b>t(2))=t(2);
    b = (b-t(1))/(t(2)-t(1));
    ImageToView(:,:,i) = reshape(b,N,M);
end

end