%%%%%%%%%%%%%%%%%%%%%%%%%%%For Reduced-Resolution%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  1) This is a test demo to show all reduced-resolution results of traditional and DL methods
%     Here, we take WV3 test dataset as example. Readers can change the corresponding director 
%     and setting to test other/your datasets
%  2) The codes of traditional methods are from the "pansharpening toolbox for distribution",
%     thus please cite the paper:
%     [1] G. Vivone, et al., A new benchmark based on recent advances in multispectral pansharpening: Revisiting
%         pansharpening with classical and emerging pansharpening methods, IEEE Geosci. Remote Sens. Mag., 
%         9(1): 53�C81, 2021
%  3) Also, if you use this toolbox, please cite our paper:
%     [2] L.-J. Deng, et al., Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks, 
%         IEEE Geosci. Remote Sens. Mag., 2022

%  LJ Deng (UESTC), 2020-02-27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: the test datasets of reduced-resolution are still too huge to upload to
% GitHub, thus we provide cloud links to readers to download them to
% successfully run this demo, including:

% i) Download link for reduced-resolution WV3-NewYork example (named "NY1_WV3_RR.mat"):
%     http:********   (put into the folder of "1_TestData/Datasets Testing")

% ii) Download link of DL's results for reduced-resolution WV3-NewYork example:
%     http:********   (put into the folder of "'2_DL_Result/WV3")

% Once you have above datasets, you can run this demo successfully, then
% understand how this demo run!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;
%% =======load directors========
% Tools
addpath([pwd,'/Tools']);

% Select algorithms to run
algorithms = {'GT','EXP','BT-H','BDSD-PC','C-GSA','SR-D',...
    'MTF-GLP-HPM-R','MTF-GLP-FS','TV','PanNet','DRPNN','MSDCNN','BDPN','DiCNN1','PNN','APNN','FusionNet'};
% director to save EPS figures for latex editing; if other dataset, please
% change the director correspondingly
satellite = 'WV3';
mat_name = 'NY1_WV3_RR';
data_name = strcat('3_EPS/', satellite, '/', mat_name);  

%% ==========Read Data and sensors' info====================
%% read the test dataset; if use your test dataset, please update in this folder
file_test = '1_TestData/Datasets Testing/NY1_WV3_RR.mat';

% get I_MS_LR, I_MS, I_PAN and sensors' info; 
load(file_test)  

% (Note: If there is no sensor's info in your dataset, 
% please find and update these info in the following commented lines):

%------ following are sensor's info for WV3 (an example for WV3)----
%     sensor = 'WV3';
%     Qblocks_size = 32;
%     bicubic = 0;% Interpolator
%     flag_cut_bounds = 1;% Cut Final Image
%     dim_cut = 21;% Cut Final Image
%     thvalues = 0;% Threshold values out of dynamic range
%     printEPS = 0;% Print Eps
%     ratio = 4;% Resize Factor
%     L = 11;% Radiometric Resolution

%% Initialization of the Matrix of Results
NumIndexes = 5;
MatrixResults = zeros(numel(algorithms),NumIndexes);
alg = 0;

% zoom-in interesting two regions of figure; you may change them
% according to your requirment
location1                = [50 70 10 30];  %default: data6: [10 50 1 60]; data7:[140 180 5 60]
location2                = [20 38 10 50];  %default: data6: [190 240 5 60]; data7:[190 235 120 150]

clear print

%% show I_MS_LR, I_GT, PAN Imgs:
if size(I_MS,3) == 4
    showImage4LR(I_MS_LR,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
else
    showImage8LR(I_MS_LR,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
end

% (Note: You may use following "showPan" without region zoom-in; otherwise, you can
% use "showPan_zoomin" for zoom-in visualization.)

%showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);
showPan_zoomin(I_PAN,printEPS,2,flag_cut_bounds,dim_cut, location1, location2);

% Note: eps figure is saved in "data_name" for latex editing
print('-depsc', strcat(data_name, '_pan', '.eps'))

%% ======GT ===================
if ismember('GT',algorithms)
    alg = alg + 1;
    [Q_avg_GT, SAM_GT, ERGAS_GT, SCC_GT_GT, Q_GT] = indexes_evaluation(I_GT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_GT,Q_avg_GT,SAM_GT,ERGAS_GT,SCC_GT_GT];
    MatrixImage(:,:,:,alg) = I_GT;
    
    % (Note: You may use following "showImage8LR" without region zoom-in; otherwise, you can
    % use "showImage8_zoomin" for zoom-in visualization.) 
    
    %showImage8LR(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_gt', '.eps'))
end

%% ======EXP ===================
if ismember('EXP',algorithms)
    alg = alg + 1;
    [Q_avg_EXP, SAM_EXP, ERGAS_EXP, SCC_GT_EXP, Q_EXP] = indexes_evaluation(I_MS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_EXP,Q_avg_EXP,SAM_EXP,ERGAS_EXP,SCC_GT_EXP];
    MatrixImage(:,:,:,alg) = I_MS;
    
    %showImage8LR(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_exp.eps'))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% CS-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====== 1) BT-H Method ======
if ismember('BT-H',algorithms)
    alg = alg + 1;
    
    cd BT-H
    t2=tic;
    I_BT_H = BroveyRegHazeMin(I_MS,I_PAN,ratio);
    time_BT_H = toc(t2);
    fprintf('Elaboration time BT-H: %.2f [sec]\n',time_BT_H);
    cd ..
    
    %%% Quality indexes computation
    [Q_avg_BT_H, SAM_BT_H, ERGAS_BT_H, SCC_GT_BT_H, Q_BT_H] = indexes_evaluation(I_BT_H,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_BT_H,Q_avg_BT_H,SAM_BT_H,ERGAS_BT_H,SCC_GT_BT_H];
    MatrixImage(:,:,:,alg) = I_BT_H;

    %showImage8LR(I_BT_H,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_BT_H,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_bth.eps'))
end

%% ====== 2) BDSD-PC Method ======
if ismember('BDSD-PC',algorithms)
    alg = alg + 1;
    
    cd BDSD
    t2=tic;
    I_BDSD_PC = BDSD_PC(I_MS,I_PAN,ratio,sensor);
    time_BDSD_PC = toc(t2);
    fprintf('Elaboration time BDSD-PC: %.2f [sec]\n',time_BDSD_PC);
    cd ..
    
    [Q_avg_BDSD_PC, SAM_BDSD_PC, ERGAS_BDSD_PC, SCC_GT_BDSD_PC, Q_BDSD_PC] = indexes_evaluation(I_BDSD_PC,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    
    MatrixResults(alg,:) = [Q_BDSD_PC,Q_avg_BDSD_PC,SAM_BDSD_PC,ERGAS_BDSD_PC,SCC_GT_BDSD_PC];
    MatrixImage(:,:,:,alg) = I_BDSD_PC;
    
    %showImage8LR(I_BDSD_PC,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_BDSD_PC,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_bdsd_pc.eps'))
end

%% ====== 3) C-GSA Method ======
if ismember('C-GSA',algorithms)
    alg = alg + 1;
    
    PS_algorithm = 'GSA'; % Pansharpening algorithm
    n_segm = 5; % Number of segments
    
    cd GS
    
    t2=tic;
    I_C_GSA = GS_Segm(I_MS,I_PAN,gen_LP_image(PS_algorithm,I_MS,I_PAN,I_MS_LR,ratio,sensor), k_means_clustering(I_MS,n_segm));
    time_C_GSA = toc(t2);
    fprintf('Elaboration time GSA: %.2f [sec]\n',time_C_GSA);
    cd ..
    
    [Q_avg_C_GSA, SAM_C_GSA, ERGAS_C_GSA, SCC_GT_C_GSA, Q_C_GSA] = indexes_evaluation(I_C_GSA,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_C_GSA,Q_avg_C_GSA,SAM_C_GSA,ERGAS_C_GSA,SCC_GT_C_GSA];
    MatrixImage(:,:,:,alg) = I_C_GSA;

    %showImage8LR(I_C_GSA,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_C_GSA,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_C_gsa.eps'))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% MRA-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====== 1) SR-D Method ======
if ismember('SR-D',algorithms)
    alg = alg + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Parameters setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TS = 7; % Tiling (dimensions of the patches are TS x TS)
    ol = 4; % Overlap (in pixels) between contiguous tile
    n_atoms = 10; % Max number of representation atoms (default value = 10)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cd SR-D
    t2=tic;
    I_SR_D = CS(I_MS,I_PAN,I_MS_LR,ratio,sensor,TS,ol,n_atoms);
    time_SR_D = toc(t2);
    fprintf('Elaboration time SR_D: %.2f [sec]\n',time_SR_D);
    cd ..
    
    [Q_avg_SR_D, SAM_SR_D, ERGAS_SR_D, SCC_GT_SR_D, Q_SR_D] = indexes_evaluation(I_SR_D,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_SR_D,Q_avg_SR_D,SAM_SR_D,ERGAS_SR_D,SCC_GT_SR_D];
    MatrixImage(:,:,:,alg) = I_SR_D;

    %showImage8LR(I_SR_D,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_SR_D,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_SR_D.eps'))
end


%% ====== 2) MTF-GLP Method ======
if ismember('MTF-GLP-HPM-R',algorithms)
    alg = alg + 1;
    
    cd GLP
    t2=tic;
    I_MTF_GLP_HPM_R = MTF_GLP_HPM_R(I_MS,I_PAN,sensor,ratio);
    time_MTF_GLP_HPM_R = toc(t2);
    fprintf('Elaboration time MTF-GLP-HPM-R: %.2f [sec]\n',time_MTF_GLP_HPM_R);
    cd ..
    
    [Q_avg_MTF_GLP_HPM_R, SAM_MTF_GLP_HPM_R, ERGAS_MTF_GLP_HPM_R, SCC_GT_MTF_GLP_HPM_R, Q_MTF_GLP_HPM_R] = indexes_evaluation(I_MTF_GLP_HPM_R,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_MTF_GLP_HPM_R,Q_avg_MTF_GLP_HPM_R,SAM_MTF_GLP_HPM_R,ERGAS_MTF_GLP_HPM_R,SCC_GT_MTF_GLP_HPM_R];
    MatrixImage(:,:,:,alg) = I_MTF_GLP_HPM_R;
    
    %showImage8LR(I_MTF_GLP_HPM_R,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_MTF_GLP_HPM_R,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_mtfglp_hpm_r.eps'))
end

%% ====== 3) MTF-GLP-FS Method ======
if ismember('MTF-GLP-FS',algorithms)
    alg = alg + 1;
    
    cd GLP
    t2=tic;
    I_MTF_GLP_FS = MTF_GLP_FS(I_MS,I_PAN,sensor,ratio);
    time_MTF_GLP_FS = toc(t2);
    fprintf('Elaboration time MTF-GLP-FS: %.2f [sec]\n',time_MTF_GLP_FS);
    cd ..
    
    [Q_avg_MTF_GLP_FS, SAM_MTF_GLP_FS, ERGAS_MTF_GLP_FS, SCC_GT_MTF_GLP_FS, Q_MTF_GLP_FS] = indexes_evaluation(I_MTF_GLP_FS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_MTF_GLP_FS,Q_avg_MTF_GLP_FS,SAM_MTF_GLP_FS,ERGAS_MTF_GLP_FS,SCC_GT_MTF_GLP_FS];
    MatrixImage(:,:,:,alg) = I_MTF_GLP_FS;
 
    %showImage8LR(I_MTF_GLP_FS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_MTF_GLP_FS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_mtfglpfs.eps'))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% VO-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====== 1) TV Method ======
if ismember('TV',algorithms)
    alg = alg + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Parameters setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    switch sensor
        case 'IKONOS'
            w=[0.1091    0.2127    0.2928    0.3854];
            c = 8;
            alpha=1.064;
            maxiter=10;
            lambda = 0.47106;
        case {'GeoEye1','WV4'}
            w=[0.1552, 0.3959, 0.2902, 0.1587];
            c = 8;
            alpha=0.75;
            maxiter=50;
            lambda = 157.8954;
        case 'WV3'
            w=[0.0657    0.1012    0.1537    0.1473    0.1245    0.1545    0.1338    0.1192];
            c = 8;
            alpha=0.75;
            maxiter=50;
            lambda = 1.0000e-03;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cd TV
    t2 = tic;
    I_TV = TV_pansharpen(I_MS_LR,I_PAN,alpha,lambda,c,maxiter,w);
    time_TV = toc(t2);
    fprintf('Elaboration time TV: %.2f [sec]\n',time_TV);
    cd ..
    
    [Q_avg_TV, SAM_TV, ERGAS_TV, SCC_GT_TV, Q_TV] = indexes_evaluation(I_TV,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_TV,Q_avg_TV,SAM_TV,ERGAS_TV,SCC_GT_TV];
    MatrixImage(:,:,:,alg) = I_TV;

    %showImage8LR(I_TV,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_TV,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_tv.eps'))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% DL-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====== 1) PanNet Method ======
% if you use other sensor's data, please update the following director and
% DL result. Note that the DL results here are obtained from our "01-DL toolbox (Pytorch)" folder, please check it.
% Similar operation for following other DL methods.


% (Note: val_bit = 2047 for 11-bit WV3, WV4 and QB data; val_bit = 1023 for 10-bit GF2 data)
val_bit  = 2047;


if ismember('PanNet',algorithms)
%     file_pannet = 'output';
    load(strcat('2_DL_Result/', satellite, '/PanNet/', 'output_', mat_name, '.mat')) 
    I_pannet = double(sr);
    alg = alg + 1;
    [Q_avg_pannet, SAM_pannet, ERGAS_pannet, SCC_pannet, Q_pannet] = indexes_evaluation(I_pannet,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_pannet,Q_avg_pannet,SAM_pannet,ERGAS_pannet,SCC_pannet];
    MatrixImage(:,:,:,alg) = I_pannet;
    
    %showImage8LR(I_pannet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_pannet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_pannet.eps'))
end

%% ====== 2) DRPNN Method ======
if ismember('DRPNN',algorithms)
%     file_drpnn = 'drpnn_wv3_rs_ny';
    load(strcat('2_DL_Result/', satellite, '/DRPNN/', 'output_', mat_name, '.mat')) 
    I_drpnn = double(sr);
    alg = alg + 1;
    [Q_avg_drpnn, SAM_drpnn, ERGAS_drpnn, SCC_drpnn, Q_drpnn] = indexes_evaluation(I_drpnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_drpnn,Q_avg_drpnn,SAM_drpnn,ERGAS_drpnn,SCC_drpnn];
    MatrixImage(:,:,:,alg) = I_drpnn;
    
    %showImage8LR(I_drpnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_drpnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_drpnn.eps'))
end

%% ====== 3) MSDCNN Method ======

if ismember('MSDCNN',algorithms)
%     file_msdcnn = 'msdcnn_wv3_rs_ny';
    load(strcat('2_DL_Result/', satellite, '/MSDCNN/', 'output_', mat_name, '.mat')) 
    I_msdcnn = double(sr);
    alg = alg + 1;
    [Q_avg_msdcnn, SAM_msdcnn, ERGAS_msdcnn, SCC_msdcnn, Q_msdcnn] = indexes_evaluation(I_msdcnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_msdcnn,Q_avg_msdcnn,SAM_msdcnn,ERGAS_msdcnn,SCC_msdcnn];
    MatrixImage(:,:,:,alg) = I_msdcnn;
    
    %showImage8LR(I_msdcnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_msdcnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_msdcnn.eps'))
end

%% ====== 4) BDPN Method ======
if ismember('BDPN',algorithms)
%     file_bdpn  = 'bdpn_wv3_rs_ny';
    load(strcat('2_DL_Result/', satellite, '/BDPN/', 'output_', mat_name, '.mat')) 
    I_bdpn  = double(sr);
    alg = alg + 1;
    [Q_avg_bdpn, SAM_bdpn, ERGAS_bdpn, SCC_bdpn, Q_bdpn] = indexes_evaluation(I_bdpn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_bdpn,Q_avg_bdpn,SAM_bdpn,ERGAS_bdpn,SCC_bdpn];
    MatrixImage(:,:,:,alg) = I_bdpn;
    
    %showImage8LR(I_bdpn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_bdpn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_bdpn.eps'))
end

%% ====== 5) DiCNN Method ======


if ismember('DiCNN1',algorithms)
%     file_dicnn = 'dicnn_wv3_rs_ny';
    load(strcat('2_DL_Result/', satellite, '/DiCNN1/', 'output_', mat_name, '.mat')) 
    I_dicnn = double(sr);
    alg = alg + 1;
    [Q_avg_dicnn, SAM_dicnn, ERGAS_dicnn, SCC_dicnn, Q_dicnn] = indexes_evaluation(I_dicnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_dicnn,Q_avg_dicnn,SAM_dicnn,ERGAS_dicnn,SCC_dicnn];
    MatrixImage(:,:,:,alg) = I_dicnn;
    
    %showImage8LR(I_dicnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_dicnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_dicnn.eps'))
end

%% ====== 6) PNN Method ======
if ismember('PNN',algorithms)
%     file_pnn = 'pnn_wv3_rs_ny';
    load(strcat('2_DL_Result/', satellite ,'/PNN/', 'output_', mat_name, '.mat')) 
    I_pnn = double(sr);
    alg = alg + 1;
    [Q_avg_pnn, SAM_pnn, ERGAS_pnn, SCC_pnn, Q_pnn] = indexes_evaluation(I_pnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_pnn,Q_avg_pnn,SAM_pnn,ERGAS_pnn,SCC_pnn];
    MatrixImage(:,:,:,alg) = I_pnn;
    
    %showImage8LR(I_pnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_pnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_pnn.eps'))
end

%% ====== 7) APNN Method ======
if ismember('APNN',algorithms)
%     file_apnn = 'apnn_wv3_rs_ny';
    load(strcat('2_DL_Result/', satellite, '/APNN/', 'output_', mat_name, '.mat')) 
    I_apnn = double(sr);
    alg = alg + 1;
    [Q_avg_apnn, SAM_apnn, ERGAS_apnn, SCC_apnn, Q_apnn] = indexes_evaluation(I_apnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_apnn,Q_avg_apnn,SAM_apnn,ERGAS_apnn,SCC_apnn];
    MatrixImage(:,:,:,alg) = I_apnn;
    
    %showImage8LR(I_apnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_apnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_apnn.eps'))
end

%% ====== 8) FusionNet Method ======

if ismember('FusionNet',algorithms)
%     file_fusionnet = 'fusionnet_wv3_rs_ny';
%     load(strcat('2_DL_Result/', satellite ,'/FusionNet/', 'output_',mat_name, '.mat'));
    load(strcat('2_DL_Result/', satellite ,'/FusionNet/', 'fusionnet_wv3_rs_ny', '.mat'));
    I_fusionnet = val_bit * double(sr);
    alg = alg + 1;
    [Q_avg_fusionnet, SAM_fusionnet, ERGAS_fusionnet, SCC_fusionnet, Q_fusionnet] = indexes_evaluation(I_fusionnet,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    MatrixResults(alg,:) = [Q_fusionnet,Q_avg_fusionnet,SAM_fusionnet,ERGAS_fusionnet,SCC_fusionnet];
    MatrixImage(:,:,:,alg) = I_fusionnet;
    
    %showImage8LR(I_fusionnet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,ratio);
    showImage8_zoomin(I_fusionnet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
    print('-depsc', strcat(data_name, '_fusionnet.eps'))
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%% Show and Save Results %%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print in LATEX
matrix2latex(MatrixResults(:,[1,3,4]),'RR_Assessment.tex', 'rowLabels',algorithms,'columnLabels',[{'Q2n'},{'SAM'},{'ERGAS'}],'alignment','c','format', '%.4f');

%% View All
if size(I_GT,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,2];
end

titleImages = algorithms;
figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

%% ======Display the final average performance =======
fprintf('\n')
disp('#######################################################')
disp(['Display the performance for:'])
disp('#######################################################')
disp(' |====Q====|===Q_avg===|=====SAM=====|======ERGAS=======|=======SCC=======')
MatrixResults

%% %%%%%%%%%%% End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
