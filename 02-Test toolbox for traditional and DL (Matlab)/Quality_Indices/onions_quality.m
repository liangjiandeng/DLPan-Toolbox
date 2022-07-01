%%%%%%%%%%%%%% Q2n aux. function
function q = onions_quality(dat1,dat2,size1)

dat1=double(dat1);
dat2=double(dat2);
dat2=cat(3,dat2(:,:,1),-dat2(:,:,2:end));
[~,~,N3]=size(dat1);
size2=size1;

% Block normalization
for i=1:N3
  [a1,s,t]=norm_blocco(squeeze(dat1(:,:,i)));
  dat1(:,:,i)=a1;
  clear a1
  if s==0
      if i==1
        dat2(:,:,i)=dat2(:,:,i)-s+1;
      else
        dat2(:,:,i)=-(-dat2(:,:,i)-s+1);   
      end
  else
      if i==1
        dat2(:,:,i)=((dat2(:,:,i)-s)/t)+1;
      else
        dat2(:,:,i)=-(((-dat2(:,:,i)-s)/t)+1);    
      end
  end
end

m1=zeros(1,N3);
m2=zeros(1,N3);

mod_q1m=0;
mod_q2m=0;
mod_q1=zeros(size1,size2);
mod_q2=zeros(size1,size2);

for i=1:N3
    m1(i)=mean2(squeeze(dat1(:,:,i)));
    m2(i)=mean2(squeeze(dat2(:,:,i)));
    mod_q1m=mod_q1m+(m1(i)^2);
    mod_q2m=mod_q2m+(m2(i)^2);
    mod_q1=mod_q1+((squeeze(dat1(:,:,i))).^2);
    mod_q2=mod_q2+((squeeze(dat2(:,:,i))).^2);
end

mod_q1m=sqrt(mod_q1m);
mod_q2m=sqrt(mod_q2m);
mod_q1=sqrt(mod_q1);
mod_q2=sqrt(mod_q2);

termine2 = (mod_q1m*mod_q2m);
termine4 = ((mod_q1m^2)+(mod_q2m^2));
int1=(size1*size2)/((size1*size2)-1)*mean2(mod_q1.^2);
int2=(size1*size2)/((size1*size2)-1)*mean2(mod_q2.^2);
termine3=int1+int2-(size1*size2)/((size1*size2)-1)*((mod_q1m^2)+(mod_q2m^2));

mean_bias=2*termine2/termine4;
if termine3==0
    q=zeros(1,1,N3);
    q(:,:,N3)=mean_bias;
else
    cbm=2/termine3;
    qu=onion_mult2D(dat1,dat2);
    
    qm=onion_mult(m1,m2);
    qv=zeros(1,N3);
    for i=1:N3
        qv(i)=(size1*size2)/((size1*size2)-1)*mean2(squeeze(qu(:,:,i)));
    end
    q=qv-(size1*size2)/((size1*size2)-1)*qm;
    
    q=q*mean_bias*cbm;
end

end