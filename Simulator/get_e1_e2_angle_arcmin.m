function [e1_e2_angle_arcmins] = get_e1_e2_angle_arcmin(e1,e2)
% Given two vectors this function returns the angle between them in arcmins. It can
% accept arrays of vectors adnd return an array of angles
%
% Inputs: - need to fill
% Ouputs: - need to fill
%
% Example usage:
%
% Revision history:
% 1.0 - 02/01/2017 - Lionel Edwin - Initial Script


for i=1:size(e1,2)
    e1_e2_angle_arcmins(i)=acosd(dot(e1(:,i)/norm(e1(:,i)),e2(:,i)/norm(e2(:,i))))*60;
end
end