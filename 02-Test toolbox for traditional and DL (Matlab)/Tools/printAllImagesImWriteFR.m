MatrixPrint(:,:,:,1) = I_MS;
MatrixPrint(:,:,:,2) = I_BT_H;
MatrixPrint(:,:,:,3) = I_BDSD;
MatrixPrint(:,:,:,4) = I_C_BDSD;
MatrixPrint(:,:,:,5) = I_BDSD_PC;
MatrixPrint(:,:,:,6) = I_GS;
MatrixPrint(:,:,:,7) = I_GSA;
MatrixPrint(:,:,:,8) = I_C_GSA;
MatrixPrint(:,:,:,9) = I_PRACS;
MatrixPrint(:,:,:,10) = I_AWLP;
MatrixPrint(:,:,:,11) = I_MTF_GLP;
MatrixPrint(:,:,:,12) = I_MTF_GLP_FS;
MatrixPrint(:,:,:,13) = I_MTF_GLP_HPM;
MatrixPrint(:,:,:,14) = I_MTF_GLP_HPM_H;
MatrixPrint(:,:,:,15) = I_MTF_GLP_HPM_R;
MatrixPrint(:,:,:,16) = I_MTF_GLP_CBD;
MatrixPrint(:,:,:,17) = I_C_MTF_GLP_CBD;
MatrixPrint(:,:,:,18) = I_MF;
MatrixPrint(:,:,:,19) = I_FE_HPM;
MatrixPrint(:,:,:,20) = I_SR_D;
MatrixPrint(:,:,:,21) = I_PWMBF;
MatrixPrint(:,:,:,22) = I_TV;
MatrixPrint(:,:,:,23) = I_RR;
MatrixPrint(:,:,:,24) = I_PNN;
MatrixPrint(:,:,:,25) = I_PNN_IDX;
MatrixPrint(:,:,:,26) = I_A_PNN;
MatrixPrint(:,:,:,27) = I_A_PNN_FT;

if size(I_MS,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,2];
end

titleImages = algorithms;

addpath([pwd,'\Tools']);

figure, MP = showImagesAll(MatrixPrint,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

cd 'Outputs'
for ii = 1 : size(MP,4)
    imwrite(MP(:,:,:,ii),sprintf('%s.png',algorithms{ii}));
end
imwrite(showPan(I_PAN,0,1,flag_cut_bounds,dim_cut),'PAN.png')
cd ..