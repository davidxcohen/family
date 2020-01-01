function eye = calcEye(gaze)
% Eye Constants
EBC2CorneaApex = 13.5; % [mm]
P2CorneaApex = 3.5; % [mm]
EBradius = 11.0; % [mm]
CorneaRadius = 8.0; % [mm]
IrisRadius = 13.0 / 2; % [mm]
PupilRadius = 2.0; % [mm]
%% Simulation conditions
EBC = [0 0 0]; % [mm] EBC
eye.EBC = EBC; 
eye.Gaze = [gaze 0]; % [deg] Gaze
P = EBC + (EBC2CorneaApex - P2CorneaApex) * [cosd(eye.Gaze(1)) sind(eye.Gaze(1)) 0];
eye.pupil = P;
P0 = EBC + (EBC2CorneaApex - P2CorneaApex) * [cosd(0) sind(0) 0]; % Optics assume to zero at nominal conditions (@ gaze = 0)
C = EBC + (EBC2CorneaApex - CorneaRadius) * [cosd(eye.Gaze(1)) sind(eye.Gaze(1)) 0];
eye.cornea = C;
eye.Led = [[27 0 0]; [27 10 0]; [27 -10 0]]; % [mm] IRLEDs 
eye.Cam = [27 -15 0]; % [mm] Camera 
eye.Eyepiece = [[30 17 0]; [30 -17 0]]; % [mm] Eyepiece corners
eye.CVG = [[ 0 0 0]; [ 8 0 0]; [ 8 -2 0]]; % [mm] converge point for the lightfield
optic_resolution = 70/400*60; % arcmin/pixel
%% Glints
C2Led = normalize2unit(eye.Led' - C'); % cornea to led unit vector
C2Cam = normalize2unit(eye.Cam' - C'); % cornea to camera unit vector
eye.glint = zeros(size(C2Cam));
for ii = 1:size(C2Led,2) % Half angle (glint) calculations
    halfAng = mean([C2Led(:,ii) C2Cam],2); % for unit vectors half angle is mean
    eye.glint(:, ii) = C' + CorneaRadius * normalize2unit(halfAng); % the glint point on cornea
end
%% contours
t = linspace(43,317,274) + gaze; % For EBC
eye.EBcontour = EBC(1:2) + EBradius * [cosd(t)' sind(t)'];
t = linspace(40,140,100) + gaze - 90; % For Cornea
eye.Ccontour = C(1:2) + CorneaRadius * [cosd(t)' sind(t)']; 
t = linspace(0,360,361) + gaze - 90; % For Modeled Cornea
eye.Ccontour_model = C(1:2) + CorneaRadius * [cosd(t)' sind(t)']; 
eye.Pcontour = [P(1:2) + PupilRadius * [cosd(eye.Gaze(1) + 90) sind(eye.Gaze(1) + 90)]; ...
            P(1:2) + IrisRadius  * [cosd(eye.Gaze(1) + 90) sind(eye.Gaze(1) + 90)]; ...
            P(1:2) - PupilRadius * [cosd(eye.Gaze(1) + 90) sind(eye.Gaze(1) + 90)]; ...
            P(1:2) - IrisRadius  * [cosd(eye.Gaze(1) + 90) sind(eye.Gaze(1) + 90)]];
%% Lightfield beam
n = -5:1:5;
contLine = 0.3;
EPpoint = (eye.Eyepiece(1,1:2) - eye.Eyepiece(2,1:2)) .* n' / length(n) + mean(eye.Eyepiece(:,1:2),1);
eye.lightfieldX = zeros([size(eye.CVG,1) length(n) 2]);
eye.lightfieldY = zeros([size(eye.CVG,1) length(n) 2]);
for jj = 1:size(eye.CVG,1)
    for ii = 1:length(n) % Lightfield beams
        eye.lightfieldX(jj, ii, :) = [EPpoint(ii,1) ((1+contLine) * eye.CVG(jj,1) - contLine * EPpoint(ii,1))];
        eye.lightfieldY(jj, ii, :) = [EPpoint(ii,2) ((1+contLine) * eye.CVG(jj,2) - contLine * EPpoint(ii,2))];
    end
end
%% pixel 
eye.Ppixel = get_e1_e2_angle_arcmin(P0'-eye.Cam', P'-eye.Cam') / optic_resolution;
eye.Gpixel(1) = get_e1_e2_angle_arcmin(P0'-eye.Cam', eye.glint(:,1)-eye.Cam') / optic_resolution;
eye.Gpixel(2) = get_e1_e2_angle_arcmin(P0'-eye.Cam', eye.glint(:,2)-eye.Cam') / optic_resolution;
eye.Gpixel(3) = get_e1_e2_angle_arcmin(P0'-eye.Cam', eye.glint(:,3)-eye.Cam') / optic_resolution;

