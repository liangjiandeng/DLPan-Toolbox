%%%%%%%%%%%%%% Q2n aux. function
function [y,a,c] = norm_blocco(x)

a=mean2(x);
c=std2(x);

if(c==0)
	c = eps;
end

y=((x-a)/c)+1;

end