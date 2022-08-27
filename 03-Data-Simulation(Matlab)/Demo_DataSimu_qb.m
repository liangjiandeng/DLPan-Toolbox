%% This is a demo to segment image into small patches (and big test imgs) 
% for the training=64x64x8 (and testing=256x256x8) pansharpening in remote sensing
% L.-J. Deng(UESTC)
% 2020-10-04
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
% please download the QB data from the website: 
% then put into the folder of "Imgs_qb"
% at last run the demo directly to get patching examples
files = dir('Imgs_qb/*.mat');  
leng = length(files);

Pre_NumInd = 1;
Pre_NumInd_test = 1;
scale = 4;

%% ----------------------------------
for i = 1:leng 
    % load inpainted images
    str = files(i).name;
    dir = strcat('load', 32, 'Imgs_qb/', str);
    eval(dir)
    
    PAN2 = I_PAN;
    LMS2 = I_MS;
    GT2  = I_GT;
    MS2  = I_MS_LR;    
    
    maxval =  max(PAN2(:));
    figure,
    subplot(2,2,1), imshow(PAN2/maxval); title('original PAN')
    subplot(2,2,2), imshow(MS2(:,:,[3 2 1])/maxval); title('LR MS')
    subplot(2,2,3), imshow(GT2(:,:,[3 2 1])/maxval); title('GT')
    subplot(2,2,4), imshow(LMS2(:,:,[3 2 1])/maxval); title('UP MS')
    
    %% leave one half of data 1 as the test data! 
    if (i==1)  % take Indianapolis to get test imags;
        cut_num = 512;
        [a, b, c] = size(GT2);
        GT        = GT2(:, cut_num+1:end,:);  % for training dataset
        GT_test   = GT2(:, 1:cut_num, :);     % for testing dataset
        
        PAN       = PAN2(:, cut_num+1:end);
        PAN_test  = PAN2(:, 1:cut_num);
        
        LMS       = LMS2(:, cut_num+1:end, :);
        LMS_test  = LMS2(:, 1:cut_num, :);
        
        MS        = MS2(:, fix(cut_num/4)+1:end, :);
        MS_test   = MS2(:, 1:fix(cut_num/4), :);
        
        %% 1) Big Test Imgs: segment pan into big Imgs 512x512x8 (testing Exm)
        size_l_test = 128; size_h_test = 512; overlap_test = 1; % (for testing data: 0<=overlaop<=64)
        
        tic
        [gt_Oneimg_test, pan_Oneimg_test, ms_Oneimg_test, lms_Oneimg_test] = segImg_new(PAN_test, LMS_test, GT_test, MS_test, size_l_test, size_h_test, scale, overlap_test);
        toc
    
        % save the Imgs into a tensor
        [NumInd_test, ~, ~, ~] = size(gt_Oneimg_test);
        Post_NumInd_test = Pre_NumInd_test + NumInd_test - 1;

        fprintf(['%d-th Img. (test):  ', 'Pre_NumInd_test = %d;  ', ' Post_NumInd_test = %d \n'], i, Pre_NumInd_test, Post_NumInd_test)
        % save data
        gt_tmp_test(Pre_NumInd_test: Post_NumInd_test, :, :, :) = gt_Oneimg_test;  % gt tensor: Nx512x512x8
        pan_tmp_test(Pre_NumInd_test: Post_NumInd_test, :, :)   = pan_Oneimg_test;  % pan tensor: Nx512x512
        ms_tmp_test(Pre_NumInd_test: Post_NumInd_test, :, :, :) = ms_Oneimg_test;  % ms tensor: Nx128x128x8
        lms_tmp_test(Pre_NumInd_test: Post_NumInd_test, :, :, :)= lms_Oneimg_test;  % lms tensor: Nx512x512x8

        Pre_NumInd_test = Post_NumInd_test + 1; 
        
    else
        GT  = GT2;
        PAN = PAN2;
        LMS = LMS2;
        MS  = MS2;
    end
    
    %% 2) small training patches (training)
    size_l = 16; size_h = 64;  overlap = 4; % (for traning data: 0<=overlaop<=16)
    
    tic
    [gt_Oneimg, pan_Oneimg, ms_Oneimg, lms_Oneimg] = segImg_new(PAN, LMS, GT, MS, size_l, size_h, scale, overlap);
    toc

    % save the patches into a tensor
    [NumInd, ~, ~, ~] = size(gt_Oneimg);
    Post_NumInd = Pre_NumInd + NumInd - 1;
    
    fprintf(['%d-th Img.(patching for training):  ', 'Pre_NumInd = %d;  ', ' Post_NumInd = %d \n'], i, Pre_NumInd, Post_NumInd)
    % save data
    gt_tmp1(Pre_NumInd: Post_NumInd, :, :, :) = gt_Oneimg;  % gt tensor: Nx64x64x8
    pan_tmp1(Pre_NumInd: Post_NumInd, :, :)   = pan_Oneimg;  % pan tensor: Nx64x64
    ms_tmp1(Pre_NumInd: Post_NumInd, :, :, :) = ms_Oneimg;  % ms tensor: Nx16x16x8
    lms_tmp1(Pre_NumInd: Post_NumInd, :, :, :)= lms_Oneimg;  % lms tensor: Nx64x64x8
            
    Pre_NumInd = Post_NumInd + 1;
    
end

%% ==========================================================
%% ==== Increase samples to 10,000 (NxCxHxW's inverse = WxHxCxN)
%% ==========================================================

exp_num = size(gt_tmp1, 1);  

if exp_num < 10000

    % Step2: two flips (lr + ud) to add examples
    gt_tmp(1:exp_num, :, :, :)             = gt_tmp1;
    gt_tmp(exp_num+1:2*exp_num, :, :, :)   = flip(gt_tmp1, 2);  % two flips (lr + ud) to add examples
    gt_tmp(2*exp_num+1:3*exp_num, :, :, :) = flip(gt_tmp1, 3);

    ms_tmp(1:exp_num, :, :, :)             = ms_tmp1;
    ms_tmp(exp_num+1:2*exp_num, :, :, :)   = flip(ms_tmp1, 2);  % two flips (lr + ud) to add examples
    ms_tmp(2*exp_num+1:3*exp_num, :, :, :) = flip(ms_tmp1, 3);

    lms_tmp(1:exp_num, :, :, :)             = lms_tmp1;
    lms_tmp(exp_num+1:2*exp_num, :, :, :)   = flip(lms_tmp1, 2);  % two flips (lr + ud) to add examples
    lms_tmp(2*exp_num+1:3*exp_num, :, :, :) = flip(lms_tmp1, 3);

    pan_tmp(1:exp_num, :, :)             = pan_tmp1;
    pan_tmp(exp_num+1:2*exp_num, :, :)   = flip(pan_tmp1, 2);  % two flips (lr + ud) to add examples
    pan_tmp(2*exp_num+1:3*exp_num, :, :) = flip(pan_tmp1, 3);

    % Step3: only select first 10000 patches for training:
    num_cut = 10000;
    gt_tmp(num_cut+1:end, :, :, :) = []; 
    ms_tmp(num_cut+1:end, :, :, :) = []; 
    lms_tmp(num_cut+1:end, :, :, :) = []; 
    pan_tmp(num_cut+1:end, :, :) = []; 
    
else
    num_cut = exp_num;
    
    gt_tmp = gt_tmp1;
    ms_tmp = ms_tmp1;
    lms_tmp=lms_tmp1;
    pan_tmp=pan_tmp1;
end

%% ==========================================================
%% (A) generate training: 1) training data (90%); 2) validation data (10%); 
%% ==========================================================
Post_NumInd = num_cut;

nz_idx    = randperm(Post_NumInd);
num_train = fix(0.9*Post_NumInd); % # training samples
num_valid  = Post_NumInd - num_train ; % # validation samples

%% ==== save to H5 file (NxCxHxW's inverse = WxHxCxN) =====
%==========================================================
%% == generate training dataset:
gt   = gt_tmp(nz_idx(1:num_train), :, :, :); % NxHxWxC=1x2x3x4
pan  = pan_tmp(nz_idx(1:num_train), :, :);   % NxHxW = 1x2x3 (PAN)
ms   = ms_tmp(nz_idx(1:num_train), :, :, :);
lms  = lms_tmp(nz_idx(1:num_train), :, :, :);

%--- for training data:
filename_train = '01-DataSimu/QB/train_qb_10000.h5';

gt   = permute(gt,[3 2 4 1]); %  beyond 2G, have to change dimension
pan_t(1,:,:,:) = pan;  % CxNxHxW = 1x2x3x4 (PAN)
pan   = permute(pan_t,[4 3 1 2]); % WxHxCxN
ms   = permute(ms,[3 2 4 1]); 
lms   = permute(lms,[3 2 4 1]); 

gtsz = size(gt);
mssz = size(ms);
lmssz = size(lms);
pansz =size(pan);


h5create(filename_train, '/gt', gtsz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_train, '/ms', mssz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_train, '/lms', lmssz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_train, '/pan', pansz(1:end), 'Datatype', 'double'); % width, height, channels, number 

h5write(filename_train, '/gt', double(gt), [1,1,1,1], size(gt));
h5write(filename_train, '/ms', double(ms), [1,1,1,1], size(ms));
h5write(filename_train, '/lms', double(lms), [1,1,1,1], size(lms));
h5write(filename_train, '/pan', double(pan), [1,1,1,1], size(pan));

clear gt ms lms pan pan_t

%% == generate validation dataset:
gt   = gt_tmp(nz_idx(num_train+1: num_train+num_valid), :, :, :);
pan  = pan_tmp(nz_idx(num_train+1: num_train+num_valid), :, :);
ms   = ms_tmp(nz_idx(num_train+1: num_train+num_valid), :, :, :);
lms  = lms_tmp(nz_idx(num_train+1: num_train+num_valid), :, :, :);

%--- for valid data:
filename_valid = '01-DataSimu/QB/valid_qb_10000.h5';

gt   = permute(gt,[3 2 4 1]); %  beyond 2G, have to change dimension
pan_t(1, :,:,:) = pan;  % NxHxWx1 = 1x2x3x4 (PAN)
pan   = permute(pan_t,[4 3 1 2]);
ms   = permute(ms,[3 2 4 1]); 
lms   = permute(lms,[3 2 4 1]); 

gtsz = size(gt);
mssz = size(ms);
pansz =size(pan);

h5create(filename_valid, '/gt', gtsz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_valid, '/ms', mssz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_valid, '/lms', gtsz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_valid, '/pan', pansz(1:end), 'Datatype', 'double'); % width, height, channels, number 

h5write(filename_valid, '/gt', double(gt), [1,1,1,1], size(gt));
h5write(filename_valid, '/ms', double(ms), [1,1,1,1], size(ms));
h5write(filename_valid, '/lms', double(lms), [1,1,1,1], size(lms));
h5write(filename_valid, '/pan', double(pan), [1,1,1,1], size(pan));

clear gt ms lms pan pan_t


%% ==========================================================
%% (B) generate Testing data:
%% ==========================================================

filename_test = '01-DataSimu/QB/TestData_qb.h5';

gt    = permute(gt_tmp_test,[3 2 4 1]); %  beyond 2G, have to change dimension
pan_t(1,:,:,:) = pan_tmp_test;  % CxNxHxW = 1x2x3x4 (PAN)
pan   = permute(pan_t,[4 3 1 2]); % WxHxCxN
ms    = permute(ms_tmp_test,[3 2 4 1]); 
lms   = permute(lms_tmp_test,[3 2 4 1]); 

gtsz  = size(gt);
mssz  = size(ms);
lmssz = size(lms);
pansz = size(pan);


h5create(filename_test, '/gt', gtsz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_test, '/ms', mssz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_test, '/lms', lmssz(1:end), 'Datatype', 'double'); % width, height, channels, number 
h5create(filename_test, '/pan', pansz(1:end), 'Datatype', 'double'); % width, height, channels, number 

h5write(filename_test, '/gt', double(gt), [1,1,1,1], size(gt));
h5write(filename_test, '/ms', double(ms), [1,1,1,1], size(ms));
h5write(filename_test, '/lms', double(lms), [1,1,1,1], size(lms));
h5write(filename_test, '/pan', double(pan), [1,1,1,1], size(pan));

clear gt ms lms pan pan_t


