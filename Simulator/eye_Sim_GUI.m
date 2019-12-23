function eye_Sim_GUI
f = figure(1);
p = uipanel(f,'Position',[0.0 0.0 0.2 0.1]);
c = uicontrol(p,'Style','slider');
c.Callback = @selection;

    function selection(src,event)
        gaze = 50 * (0.5 - c.Value); % c.Value is between 0-->1
        eye = calcEye(gaze); 
        plotEye(eye)
    end

end