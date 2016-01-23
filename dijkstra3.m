function dijkstra3 ()

maxValue=28; % infinity value
a=maxValue+zeros(21,21);
% track=-2 just for the graphics
track=-2;
a(1,1)=track;

% goals
a(3,3)=0;
a(3,7)=0;

% barriers - walls
a(10,6:18)=-1;
a(10:18,6)=-1;
a(10:18,18)=-1;
a(18,6:16)=-1;
a(12,8:14)=-1;
a(12:14,8)=-1;
a(14,6:8)=-1;

% player
player=[13 7];

iterations=maxValue-1;
for kk=1:iterations
    for ii=3:2:19
        for jj=3:2:19
            if a(ii-1,jj)~= -1 && a(ii-2,jj) == kk-1 && a(ii,jj) == maxValue
                a(ii,jj)=kk;
            else if a(ii+1,jj)~= -1 && a(ii+2,jj) == kk-1 && a(ii,jj) == maxValue
                    a(ii,jj)=kk;
                else if a(ii,jj-1)~= -1 && a(ii,jj-2) == kk-1 && a(ii,jj) == maxValue
                        a(ii,jj)=kk;
                    else if a(ii,jj+1)~= -1 && a(ii,jj+2) == kk-1 && a(ii,jj) == maxValue 
                            a(ii,jj)=kk;
                        end
                    end
                end
            end
        end
    end
    % delay
    delay(0.3);
    % draw results
    figure(1);
    imagesc(a);colorbar;drawnow
end
b1=a;

% define place and value of the player
ii=player(1);jj=player(2);
distance=a(ii,jj);

% begining of the course back
b1(ii,jj)=track;
figure(1);imagesc(b1);colorbar

% way back: finding neighbor without wall that has a smaller value as 
% a next step  
while distance~=0
    if a(ii-2,jj)==(distance - 1) && a(ii-1,jj)~= -1
        ii=ii-2;
    else if a(ii+2,jj)==(distance - 1) && a(ii+1,jj)~= -1
            ii=ii+2;
        else if a(ii,jj-2)==(distance - 1) && a(ii,jj-1)~= -1 
                jj=jj-2;
            else 
                jj=jj+2;
            end
        end
    end
    distance=distance-1;
    b1(ii,jj)=track;
    disp([num2str(ii) ';' num2str(jj)]);
    % show results
    imagesc(b1);colorbar;drawnow
    % delay
    delay(.3);
end

function delay(delay_seconds)

tic;

while toc < delay_seconds

end
