function [gt, pan, ms, lms] = segImg_new(PAN, LMS, GT, MS, size_l, size_high, scale, overlap)
% This is a core function to segment big images into small patches
% LJ Deng (UESTC); 2020-10-09

% PAN:       big PAN image
% LMS:       big upsampled MS image
% GT:        big original HRMS image
% MS:        big original LR MS image
% size_l:    the patch size LR patch
% size_high: the patch size HR patch
% scale:     spatial ration of PAN and MS, here, scale = 4
% overlap:   the overlap among segmented patches
% gt:        segmented ground-truth (gt) or labeled data 
% pan:       segmented pan data
% ms:        segmented ms data
% lms:       segmented lms data

%% --------------------------
[h, w, c]   = size(MS);   % size of LR: 
H = scale*h; W = scale*w;

size_low    = size_l; % patch size of LR: 16x16
size_h      = size_high; % patch size of LR: 64x64
overlap_low = overlap;  %  overlap of LR
overlap_h   = scale*overlap; % overlap of HR

% set patch indexs 
%---- LR indexs ---------
gridy = 1:size_low - overlap_low : w;%-(mod(w,size_low-overlap_low)+1+size_low-overlap_low);
gridy((gridy+size_low-1) > w) = [];  % delet boudary points
gridx = 1:size_low - overlap_low: h;%-(mod(h,size_low-overlap_low)+1+size_low-overlap_low);
gridx((gridx+size_low-1) > h) = [];  % delet boudary points

%---- HR indexs ---------
Gridy = 1:size_h - overlap_h : W;%-(mod(W,size_h-overlap_h)+1+size_h-overlap_h);   % is 2 or 8? ===>must be some problem here!
Gridy((Gridy+size_h-1) > W) = [];  % delet boudary points
Gridx = 1:size_h - overlap_h : H;%-(mod(H,size_h-overlap_h)+1+size_h-overlap_h);
Gridx((Gridx+size_h-1) > H) = [];  % delet boudary points

%% -----Pre-define variables' sizes--------
pan  = zeros(size(gridx,2)*size(gridy,2), size_h, size_h);
lms  = zeros(size(gridx,2)*size(gridy,2), size_h, size_h, c);
gt  = zeros(size(gridx,2)*size(gridy,2), size_h, size_h, c);
ms  = zeros(size(gridx,2)*size(gridy,2), size_low, size_low, c);

%% -----loops to segment--------
cnt = 0;
Num = 0;
for i = 1: length(gridx)
    for j = 1:length(gridy)
        cnt = cnt + 1;
        Num = Num + 1;
        xx = gridx(i);
        yy = gridy(j);
        XX = Gridx(i);
        YY = Gridy(j);    
        
        % ---start to segment------
        pan_p = PAN(XX:XX+size_h-1, YY:YY+size_h-1);% 64x64: signle pan patch
        pan(Num, :, :) = pan_p; % save single to a "pan" tensor: Nx64x64

        lms_p = LMS(XX:XX+size_h-1, YY:YY+size_h-1, :); % 64x64x8: signle lms patch
        lms(Num, :, :, :) = lms_p; % save single to a "lms" tensor: Nx64x64x8

        gt_p = GT(XX:XX+size_h-1, YY:YY+size_h-1, :); % 64x64x8: signle gt patch
        gt(Num, :, :, :) = gt_p; % save single to a "gt" tensor: Nx64x64x8
        
        ms_p  = MS(xx:xx+size_low-1, yy:yy+size_low-1, :); % 16x16x8: signle ms patch      
        ms(Num, :, :, :) = ms_p; % save single to a "ms" tensor: Nx16x16x8
              
        if Num == 1  % to see if there needs registration!
            maxval =  max(PAN(:));
            ww(:,:,1)=gt_p(:,:,3);  % gt
            ww(:,:,2)=gt_p(:,:,2);
            ww(:,:,3)=gt_p(:,:,1);
            kk(:,:,1)=lms_p(:,:,3);  % gt
            kk(:,:,2)=lms_p(:,:,2);
            kk(:,:,3)=lms_p(:,:,1);
            pp       = pan_p;   % pan        
            figure,
            subplot(1,3,1), imshow(double(ww)/maxval + 0.3); title('gt')
            subplot(1,3,2), imshow(double(pp)/maxval + 0.3); title('pan')
            subplot(1,3,3), imshow(double(kk)/maxval + 0.3); title('lms')
        end
        
     end
end
%% -----End loops--------

end

