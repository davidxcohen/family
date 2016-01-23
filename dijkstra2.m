function dijkstra2 ()

maxValue=30; % infinity value
a=maxValue+zeros(21,21);
alreadyModified=(a==1);
alreadyModified(2:2:20,:)=1;
alreadyModified(:,2:2:20)=1;
alreadyModified(1,:)=1;
alreadyModified(21,:)=1;
alreadyModified(:,1)=1;
alreadyModified(:,21)=1;
%figure(1);imagesc(alreadyModified)
barrier=(a==1);


% goals
a(3,3)=0;
alreadyModified(3,3)=1;
a(3,7)=0;
alreadyModified(3,7)=1;

% barriers - walls
barrier(10,6:18)=1;
barrier(10:18,6)=1;
barrier(10:18,18)=1;
barrier(18,6:14)=1;
barrier(12,8:14)=1;
barrier(12:14,8)=1;
barrier(14,6:8)=1;
barrier(18,16:18)=1;
% player
player=[13 7];

% neighbor mask
% mask=[0 1 0;1 0 1;0 1 0];
% mask=(mask==1);

b=a;
iterations=maxValue;
for kk=1:iterations
    b1=b;
    for ii=3:2:19
        for jj=3:2:19
            % finding the minimal neighbor value d, if the is no barrier
            subB=b(ii-2:2:ii+2,jj-2:2:jj+2);
            subBarier=barrier(ii-1:ii+1,jj-1:jj+1);
%            d=min(subB(mask & ~subBarier));
            d=maxValue;
            if ~subBarier(1,2) && subB(1,2) < d
                d=subB(1,2);
            else if ~subBarier(3,2) && subB(3,2) < d
                    d=subB(3,2);
                else if ~subBarier(2,1) && subB(2,1) < d
                        d=subB(2,1);
                    else if ~subBarier(2,3) && subB(2,3) < d
                            d=subB(2,3);
                        end
                    end
                end
            end

            if (d~=maxValue && ~alreadyModified(ii,jj))
                % If there is a "non-infinity" neighbor, the current point
                % become d+1
                b1(ii,jj)=1+d;
                alreadyModified(ii,jj)=1;
            end
        end

    end
    b=b1;
    % delay
    delay(1);
    % draw results
    figure(1);
    imagesc(b+2*barrier);colorbar;drawnow
end

% define place and value of the player
ii=player(1);jj=player(2);
distance=b(ii,jj);

% begining of the course back
b1(ii,jj)=0;
imagesc(b1+2*barrier);colorbar

% way back: finding neighbor without wall that has a smaller value as 
% a next step  
while distance~=0
    if b(ii-2,jj)==(distance - 1) && ~barrier(ii-1,jj)
        ii=ii-2;
    else if b(ii+2,jj)==(distance - 1) && ~barrier(ii+1,jj)
            ii=ii+2;
        else if b(ii,jj-2)==(distance - 1) && ~barrier(ii,jj-1) 
                jj=jj-2;
            else 
                jj=jj+2;
            end
        end
    end
    distance=distance-1;
    b1(ii,jj)=0;
    
    % show results
    imagesc(b1+2*barrier);colorbar;drawnow
    % delay
    delay(1);
end

function delay(delay_seconds)

tic;

while toc < delay_seconds

end
