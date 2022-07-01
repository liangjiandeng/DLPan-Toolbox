function ent=rectangleonimage(pic,sw,n, ch, c, scale, type)
% sw: the location of the up-left, down-right
% n: the width of the line
% ch: ch = 1 (gray image); ch = 3 (color image) 
% c: the color of the line: c=1(red); c=2(green); c=3(blue);c=others
% scale: the salce of zooming in for SR
% type =1 (put to down-left); type =2 (put to down-right); 
% type =3 (put to up-right); type =4 (put to up-left); 
% Liang-Jian Deng (UESTC)
% improved time: 2017-3-11
%==============================%

if nargin< 5
    scale = [];
end
x0=sw(1);x1=sw(2);y0=sw(3);y1=sw(4);
[p q ch]=size(pic);

max_val = 1;

%ch=1:gray image; ch=3: color image
if ch==1
    if c==1
        pic(x0:x1,y0:y0+n)=max_val;
        pic(x0:x1,y1-n:y1)=max_val;
        pic(x0:x0+n,y0:y1)=max_val;
        pic(x1-n:x1,y0:y1)=max_val;
    elseif c==2
        pic(x0:x1,y0:y0+n)=0;
        pic(x0:x1,y1-n:y1)=0;
        pic(x0:x0+n,y0:y1)=0;
        pic(x1-n:x1,y0:y1)=0;
    else
        pic(x0:x1,y0:y0+n)=max_val-pic(x0:x1,y0:y0+n); %È¡·´
        pic(x0:x1,y1-n:y1)=max_val- pic(x0:x1,y1-n:y1);
        pic(x0:x0+n,y0:y1)=max_val-pic(x0:x0+n,y0:y1);
        pic(x1-n:x1,y0:y1)=max_val-pic(x1-n:x1,y0:y1);
    end
end

if ch==3
    if c==1
        pic(x0:x1,y0:y0+n,1)=max_val; pic(x0:x1,y0:y0+n,2)=0; pic(x0:x1,y0:y0+n,3)=0;
        pic(x0:x1,y1-n:y1,1)=max_val;   pic(x0:x1,y1-n:y1,2)=0;   pic(x0:x1,y1-n:y1,3)=0;
        pic(x0:x0+n,y0:y1,1)=max_val; pic(x0:x0+n,y0:y1,2)=0; pic(x0:x0+n,y0:y1,3)=0;
        pic(x1-n:x1,y0:y1,1)=max_val;   pic(x1-n:x1,y0:y1,2)=0;   pic(x1-n:x1,y0:y1,3)=0;
        
    elseif c==2
        pic(x0:x1,y0:y0+n,1)=0;pic(x0:x1,y0:y0+n,2)=max_val;pic(x0:x1,y0:y0+n,3)=0;
        pic(x0:x1,y1-n:y1,1)=0;pic(x0:x1,y1-n:y1,2)=max_val;pic(x0:x1,y1-n:y1,3)=0;
        pic(x0:x0+n,y0:y1,1)=0;pic(x0:x0+n,y0:y1,2)=max_val;pic(x0:x0+n,y0:y1,3)=0;
        pic(x1-n:x1,y0:y1,1)=0;pic(x1-n:x1,y0:y1,2)=max_val;pic(x1-n:x1,y0:y1,3)=0;

    elseif c==3   
        pic(x0:x1,y0:y0+n,1)=0;pic(x0:x1,y0:y0+n,2)=0;pic(x0:x1,y0:y0+n,3)=max_val;
        pic(x0:x1,y1-n:y1,1)=0;pic(x0:x1,y1-n:y1,2)=0;pic(x0:x1,y1-n:y1,3)=max_val;
        pic(x0:x0+n,y0:y1,1)=0;pic(x0:x0+n,y0:y1,2)=0;pic(x0:x0+n,y0:y1,3)=max_val;
        pic(x1-n:x1,y0:y1,1)=0;pic(x1-n:x1,y0:y1,2)=0;pic(x1-n:x1,y0:y1,3)=max_val;

    else                          %inverse
        pic(x0:x1,y0:y0+n,1:3)=max_val-pic(x0:x1,y0:y0+n,1:3);
        pic(x0:x1,y1-n:y1,1:3)=max_val-pic(x0:x1,y1-n:y1,1:3);
        pic(x0:x0+n,y0:y1,1:3)=max_val-pic(x0:x0+n,y0:y1,1:3);
        pic(x1-n:x1,y0:y1,1:3)=max_val-pic(x1-n:x1,y0:y1,1:3);
    end
end

ent=pic; 
sampIm = pic(x0:x1, y0:y1, :);
SampIm = imresize(sampIm, scale,'nearest'); % nearest to zooming in the local part
switch type
    case 1   %  put zoom in image on the down-left
        [a, b, third] = size(SampIm);
        ent((p-a+1):p,1:b, :) = SampIm;
    case 2  %  put zoom in image on the down-left
        [a, b, third] = size(SampIm);
        ent((p-a+1):p,(q-b+1):q, :) = SampIm;
        
    case 3  %  put zoom in image on the up-right
        [a, b, third] = size(SampIm);
        ent(1:a,(q-b+1):q, :) = SampIm;
        
    case 4  %  put zoom in image on the up-right
        [a, b, third] = size(SampIm);
        ent(1:a,1:b, :) = SampIm;        
end




