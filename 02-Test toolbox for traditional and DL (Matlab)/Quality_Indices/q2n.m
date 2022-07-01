%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Q2n index. 
% 
% Interface:
%           [Q2n_index, Q2n_index_map] = q2n(I_GT, I_F, Q_blocks_size, Q_shift)
%
% Inputs:
%           I_GT:               Ground-Truth image;
%           I_F:                Fused Image;
%           Q_blocks_size:      Block size of the Q-index locally applied;
%           Q_shift:            Block shift of the Q-index locally applied.
%
% Outputs:
%           Q2n_index:          Q2n index;
%           Q2n_index_map:      Map of Q2n values.
%
% References:
%           [Garzelli09]        A. Garzelli and F. Nencini, "Hypercomplex quality assessment of multi/hyper-spectral images," 
%                               IEEE Geoscience and Remote Sensing Letters, vol. 6, no. 4, pp. 662–665, October 2009.
%           [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                               IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q2n_index, Q2n_index_map] = q2n(I_GT, I_F, Q_blocks_size, Q_shift)

[N1,N2,N3]=size(I_GT);
size2=Q_blocks_size;

stepx=ceil(N1/Q_shift);
stepy=ceil(N2/Q_shift);

if stepy<=0
    stepy=1;
    stepx=1;
end

est1=(stepx-1)*Q_shift+Q_blocks_size-N1;
est2=(stepy-1)*Q_shift+Q_blocks_size-N2;

if sum([(est1~=0),(est2~=0)])>0
  refref=[];
  fusfus=[];
  
  for i=1:N3
      a1=squeeze(I_GT(:,:,1));
    
      ia1=zeros(N1+est1,N2+est2);
      ia1(1:N1,1:N2)=a1;
      ia1(:,N2+1:N2+est2)=ia1(:,N2:-1:N2-est2+1);
      ia1(N1+1:N1+est1,:)=ia1(N1:-1:N1-est1+1,:);
      refref=cat(3,refref,ia1);
      
      if i<N3
          I_GT=I_GT(:,:,2:end);
      end
  end

  I_GT=refref;
  clear refref
  
  for i=1:N3
      a2=squeeze(I_F(:,:,1));
      
      ia2=zeros(N1+est1,N2+est2);
      ia2(1:N1,1:N2)=a2;
      ia2(:,N2+1:N2+est2)=ia2(:,N2:-1:N2-est2+1);
      ia2(N1+1:N1+est1,:)=ia2(N1:-1:N1-est1+1,:);
      fusfus=cat(3,fusfus,ia2);
      
      if i<N3
          I_F=I_F(:,:,2:end);
      end
  end
  
  I_F=fusfus;
  clear fusfus a1 a2 ia1 ia2

end

I_F=uint16(I_F);
I_GT=uint16(I_GT);

[N1,N2,N3]=size(I_GT);

if ((ceil(log2(N3)))-log2(N3))~=0
    Ndif=(2^(ceil(log2(N3))))-N3;
    dif=zeros(N1,N2,Ndif);
    dif=uint16(dif);
    I_GT=cat(3,I_GT,dif);
    I_F=cat(3,I_F,dif);
end
[~,~,N3]=size(I_GT);

valori=zeros(stepx,stepy,N3);

for j=1:stepx
    for i=1:stepy
        o=onions_quality(I_GT(((j-1)*Q_shift)+1:((j-1)*Q_shift)+Q_blocks_size,((i-1)*Q_shift)+1:((i-1)*Q_shift)+size2,:),I_F(((j-1)*Q_shift)+1:((j-1)*Q_shift)+Q_blocks_size,((i-1)*Q_shift)+1:((i-1)*Q_shift)+size2,:),Q_blocks_size);
        valori(j,i,:)=o;    
    end
end

Q2n_index_map=sqrt(sum((valori.^2),3));

Q2n_index=mean2(Q2n_index_map);

end