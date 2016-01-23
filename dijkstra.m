a=zeros(11,11);
alreadyModified=(a==1);
a=a+10;

% goals
a(2,2)=0;
alreadyModified(2,2)=1;
a(2,4)=0;
alreadyModified(2,4)=1;
% 
% barriers
alreadyModified(5,3:8)=1;
alreadyModified(7,3:8)=1;

mask=[0 1 0;1 0 1;0 1 0];
mask=(mask==1);
b=a;
iterations=13;
for kk=1:iterations
    b1=b;
    for ii=2:10
        for jj=2:10
            subB=b(ii-1:ii+1,jj-1:jj+1);
            d=min(subB(mask));
            if (d~=10 && ~alreadyModified(ii,jj))
                b1(ii,jj)=1+d;
                alreadyModified(ii,jj)=1;
            end
        end
    end
    b=b1
    figure(1);imagesc(b);colorbar
    alreadyModified
end
