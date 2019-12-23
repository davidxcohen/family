function plotEye(eye)
%% figure
figure(1)
% f = figure;
% p = uipanel(f,'Position',[0.1 0.1 0.9 0.9]);
% c = uicontrol(p,'Style','slider');
% c.Value = 0.5;
% eyeball
plot(eye.EBcontour(:, 1), eye.EBcontour(:, 2), 'b', 'linewidth', 2); hold on
% eyepiece
plot(eye.Eyepiece(:,1)+0.0,eye.Eyepiece(:,2), 'g', 'linewidth', 4);
plot(eye.Eyepiece(:,1)+0.4,eye.Eyepiece(:,2), 'b', 'linewidth', 4);
plot(eye.Eyepiece(:,1)+0.8,eye.Eyepiece(:,2), 'r', 'linewidth', 4);
% Pupil contours
plot(eye.Pcontour(3:4,1), eye.Pcontour(3:4,2), 'k', 'linewidth',3)
plot(eye.Pcontour(1:2,1), eye.Pcontour(1:2,2), 'k', 'linewidth',3)
% P, C, EBC
scatter([eye.pupil(1) eye.cornea(1) eye.EBC(1)], [eye.pupil(2) eye.cornea(2) eye.EBC(2)], 100, '.k') 
text(eye.pupil(1),eye.pupil(2),'P ', 'HorizontalAlignment', 'right');
text(eye.cornea(1),eye.cornea(2),'C ', 'HorizontalAlignment', 'right');
text(eye.EBC(1),eye.EBC(2),'EBC ', 'HorizontalAlignment', 'right');
% IRled
scatter(eye.Led(:,1),eye.Led(:,2), 300,'or','filled')
scatter(eye.Led(:,1)+1,eye.Led(:,2), 500,'sr','filled')
% Camera
scatter(eye.Cam(1)+1.5,eye.Cam(2),500, 'sb','filled')
scatter(eye.Cam(1),eye.Cam(2),300, '>b','filled')
% lightfield beams
col =   'kcmg';
shape = ':--:';
for jj = 1:size(eye.CVG,1)
    for ii = 1:size(eye.lightfieldX,2) % Lightfield beams
        plot(squeeze(eye.lightfieldX(jj, ii,:)), squeeze(eye.lightfieldY(jj, ii,:)),[col(jj) shape(jj)]);
    end
end
% Glint line
for ii = 1:size(eye.Led,1)
    plot([eye.Led(ii, 1) eye.glint(1, ii) eye.Cam(1)], [eye.Led(ii, 2) eye.glint(2, ii) eye.Cam(2)], 'r', 'linewidth',2);
end
% Cornea
plot(eye.Ccontour_model(:, 1), eye.Ccontour_model(:, 2), ':b', 'linewidth', 1); 
plot(eye.Ccontour(:, 1), eye.Ccontour(:, 2), 'b', 'linewidth', 2); hold off
grid on
axis equal

text(-10, 15, sprintf('Gaze = %1.1f˚ \nGlint pixels = %1.1f, %1.1f, %1.1f\nPupil pixel = %1.1f', eye.Gaze(1), eye.Gpixel(1),eye.Gpixel(2),eye.Gpixel(3), eye.Ppixel))