function dijkstra4 ()
% split into separated functions

maxValue=28; % infinity value
a=maxValue+zeros(21,21);
% track=-2 just for the graphics
track=-2;
a(1,1)=track;

%% initialize the board
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

%% distribute the "energy" 
a=f4(a,maxValue);

%% calculate the path back
% b1 is matrix for drawing the path back
b1=a;

% define place and value of the player
ii=player(1);jj=player(2);
distance=a(ii,jj);

% begining of the course back
b1(ii,jj)=track;
figure(1);imagesc(b1);colorbar

% way back: finding neighbor without wall that has a smaller value as 
% a next step 
pathBack=[ii jj];
while distance~=0
    [ii jj]=f6(a,ii,jj,distance);
    distance=distance-1;
    pathBack=[pathBack ; [ii jj]]

    % graphics
    b1(ii,jj)=track;
    printAr(b1);
end
disp(pathBack)
end

%% function collection
function delay(delay_seconds)

tic;

while toc < delay_seconds

end
end

function funcResults=myFor(startInd,stepInd,endInd,fhandle)
f = fcnchk(fhandle)
for ii=startInd:stepInd:endInd
    funcResults=fhandle(ii);
end
end

function printAr(a)
    % delay
    delay(0.1);
    % draw results
    figure(1);
    imagesc(a);colorbar;drawnow
end

function a=f1a(a,ii,jj,kk,maxValue)
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

function a=f1(a,ii,jj,kk,maxValue)
if a(ii-1,jj)~= -1 && a(ii-2,jj) == kk-1 && a(ii,jj) == maxValue
    a(ii,jj)=kk; end
if a(ii+1,jj)~= -1 && a(ii+2,jj) == kk-1 && a(ii,jj) == maxValue
    a(ii,jj)=kk; end
if a(ii,jj-1)~= -1 && a(ii,jj-2) == kk-1 && a(ii,jj) == maxValue
    a(ii,jj)=kk; end
if a(ii,jj+1)~= -1 && a(ii,jj+2) == kk-1 && a(ii,jj) == maxValue 
    a(ii,jj)=kk; end
end

function a=f2(a,ii,kk,maxValue)
for jj=3:2:19
    a=f1(a,ii,jj,kk,maxValue);
end
end

function a=f3(a,kk,maxValue)
for ii=3:2:19
    a=f2(a,ii,kk,maxValue);
end
end

function a=f4(a,maxValue)
for kk=1:1:maxValue-1
    a=f3(a,kk,maxValue);
    printAr(a);
end
end

function [ii jj]=f5(a,ii,jj,distance)
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
end

function [ii jj]=f6(a,ii,jj,distance)
    if a(ii-2,jj)==(distance - 1) && a(ii-1,jj)~= -1
        ii=ii-2; end
    if a(ii+2,jj)==(distance - 1) && a(ii+1,jj)~= -1
        ii=ii+2; end
    if a(ii,jj-2)==(distance - 1) && a(ii,jj-1)~= -1 
        jj=jj-2; end
    if a(ii,jj+2)==(distance - 1) && a(ii,jj+1)~= -1 
        jj=jj+2; end
end