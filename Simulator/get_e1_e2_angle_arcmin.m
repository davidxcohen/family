function [e1_e2_angle_arcmins] = get_e1_e2_angle_arcmin(e1,e2)
for i=1:size(e1,2)
    e1_e2_angle_arcmins(i)=acosd(dot(e1(:,i)/norm(e1(:,i)),e2(:,i)/norm(e2(:,i))))*60;
end
end
