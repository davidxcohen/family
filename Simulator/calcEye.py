import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# plt.style.use('dark_background')
# plt.style.use('seaborn-dark-palette')


def normalize2unit(input):
    output = input**2
    # output = input ./ repmat(sqrt(sum(input.^2)),3,1)
    return output


def normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/numpy.linalg.norm(x, ord=2, axis=1, keepdims=True)


def calcEye(gaze=5.):
    # Eye Constants
    gazeRAD = [np.radians(gaze), 0]
    EBC2CorneaApex = 13.5  # [mm]
    P2CorneaApex = 3.5  # [mm]
    EBradius = 12.0  # [mm]
    CorneaRadius = 8.0  # [mm]
    IrisRadius = 13.0 / 2  # [mm]
    PupilRadius = 2.0  # [mm]

    # Simulation conditions
    EBC = np.array([0, 0, 0])  # [mm] EBC
    P = EBC + (EBC2CorneaApex - P2CorneaApex) * np.array([np.cos(gazeRAD[0]), np.sin(gazeRAD[0]), 0])
    P0 = EBC + (EBC2CorneaApex - P2CorneaApex) * np.array([np.cos(0), np.sin(0), 0])
                                                        # Optics assume to zero at nominal conditions (@ gazeRAD = 0)
    C = EBC + (EBC2CorneaApex - CorneaRadius) * np.array([np.cos(gazeRAD[0]), np.sin(gazeRAD[0]), 0])

    eye = {'EBC': EBC,
           'Gaze': np.array([gazeRAD, 0]),
           'pupil': P,
           'cornea': C,
           'Led': np.array([[27, 0, 0],
                            [27, 10, 0],
                            [27, 5, 0],
                            [27, -10, 0]]),
           'Cam': np.array([27, -15, 0]),
           'Eyepiece': np.array([[30, 17, 0], [30, -17, 0]]),  # [mm] Eyepiece corners
           'CVG': np.array([[30, 17, 0], [30, -17, 0]]),  # [mm] converge point for the lightfield
           }

    optic_resolution = 70. / 400. * 60.  # arcmin/pixel

    # glint
    C2Led = normalize_rows(eye['Led'] - C)  # cornea to led unit vector
    C2Cam = normalize(eye['Cam'] - C)  # cornea to Camera unit vector
    C2Cam = numpy.matlib.repmat(C2Cam, np.shape(C2Led)[0], 1)
    # halfAng= np.zeros(np.shape(C2Led))  #
    # for ii in range(np.shape(C2Led)[1]):  # Half angle (glint) calculations (ii index for x, y, z)
    halfAng = np.mean([C2Led, C2Cam], axis=0)  # for unit vectors half angle is mean
    eye['glint'] = C + CorneaRadius * normalize_rows(halfAng)  # the glint point on cornea

    #  contours
    t = np.radians(np.linspace(43, 317, 274)) + gazeRAD[0]  # For EBC
    eye['EBcontour'] = numpy.matlib.repmat(EBC[:2], len(t), 1) + EBradius * np.array([np.cos(t), np.sin(t)]).T
    t = np.radians(np.linspace(55, 125, 7)) + gazeRAD[0] - np.radians(90)  # For Cornea
    eye['Ccontour'] = numpy.matlib.repmat(C[:2], len(t), 1) + CorneaRadius * np.array([np.cos(t), np.sin(t)]).T
    t = np.radians(np.linspace(0, 360, 361)) + gazeRAD[0] - np.radians(90)  # For Modeled Cornea
    eye['Ccontour_model'] = numpy.matlib.repmat(C[:2], len(t), 1) + CorneaRadius * np.array([np.cos(t), np.sin(t)]).T
    eye['Pcontour'] = np.array([
                    (P[:2] + PupilRadius * np.array([np.cos(gazeRAD[0] + np.radians(90)), np.sin(gazeRAD[0] + np.radians(90))])),
                    (P[:2] + IrisRadius * np.array([np.cos(gazeRAD[0] + np.radians(90)), np.sin(gazeRAD[0] + np.radians(90))])),
                    (P[:2] - PupilRadius * np.array([np.cos(gazeRAD[0] + np.radians(90)), np.sin(gazeRAD[0] + np.radians(90))])),
                    (P[:2] - IrisRadius * np.array([np.cos(gazeRAD[0] + np.radians(90)), np.sin(gazeRAD[0] + np.radians(90))]))])
    Cam = numpy.matlib.repmat(eye['Cam'], np.shape(C2Led)[0], 1)
    # eye['glint_beam'] = np.concatenate((Cam, eye['glint'], eye['Led']))
    eye['glint_beam'] = np.array([Cam, eye['glint'], eye['Led']])
    eye['limbus'] = np.array([eye['EBcontour'][-1], eye['Ccontour'][0], eye['Ccontour'][-1], eye['EBcontour'][0]])

    # Lightfield beams
    n = np.linspace(-5, 5, 7)
    contLine = 0.3
    EPpoint = numpy.matlib.repmat(eye['Eyepiece'][0] - eye['Eyepiece'][-1], len(n), 1) * \
              numpy.matlib.repmat(n, 3, 1).T / len(n) + \
              numpy.matlib.repmat(np.mean(eye['Eyepiece'], axis=0), len(n), 1)  # Points on the eyepiece where the beam emitted
    eye['lightfieldX'] = np.zeros([np.shape(eye['CVG'][1])[0] - 1, len(n), 2])
    eye['lightfieldY'] = np.zeros([np.shape(eye['CVG'][1])[0] - 1, len(n), 2])
    for ii in range(np.shape(eye['CVG'][0])[0] - 1):  # convergence points
        for jj in range(len(n)):  # Lightfield beams
            # print('ii,jj: ', ii, jj)
            eye['lightfieldX'][ii, jj, :] = np.array([EPpoint[jj, 0],
                                                      (1+contLine) * eye['CVG'][ii, 0] - contLine * EPpoint[jj, 0]])
            eye['lightfieldY'][ii, jj, :] = np.array([EPpoint[jj, 1],
                                                      (1+contLine) * eye['CVG'][ii, 1] - contLine * EPpoint[jj, 1]])
    # print('lightfieldX: ', eye['lightfieldX'])

    # %% pixel
    # eye.Ppixel = get_e1_e2_angle_arcmin(P0'-eye.Cam', P'-eye.Cam') / optic_resolution;
    # eye.Gpixel(1) = get_e1_e2_angle_arcmin(P0'-eye.Cam', eye.glint(:,1)-eye.Cam') / optic_resolution;
    # eye.Gpixel(2) = get_e1_e2_angle_arcmin(P0'-eye.Cam', eye.glint(:,2)-eye.Cam') / optic_resolution;
    # eye.Gpixel(3) = get_e1_e2_angle_arcmin(P0'-eye.Cam', eye.glint(:,3)-eye.Cam') / optic_resolution;
    return eye


def plotEye(eye):
    # fig, ax = plt.subplots()
    # plt.subplots_adjust(bottom=0.25)  # leave space for the slider
    # plt.figure()
    ax.plot(eye['Ccontour'].T[0], eye['Ccontour'].T[1], 'b', linewidth=2, markersize=12)
    ax.plot(eye['Ccontour_model'].T[0], eye['Ccontour_model'].T[1], ':b', linewidth=2, markersize=12)  # continue the
                                                                                    # cornea with dotted line
    ax.plot(eye['EBcontour'].T[0], eye['EBcontour'].T[1], 'b', linewidth=2, markersize=12)
    ax.plot(eye['limbus'].T[0][0:2], eye['limbus'].T[1][0:2], 'g', linewidth=2, markersize=12)  # Bottom limbus
    ax.plot(eye['limbus'].T[0][2:4], eye['limbus'].T[1][2:4], 'g', linewidth=2, markersize=12)  # top limbus
    ax.plot(eye['Pcontour'].T[0][0:2], eye['Pcontour'].T[1][0:2], 'c', linewidth=4, markersize=12)
    ax.plot(eye['Pcontour'].T[0][2:4], eye['Pcontour'].T[1][2:4], 'c', linewidth=4, markersize=12)
    ax.plot(eye['Eyepiece'].T[0]  , eye['Eyepiece'].T[1], 'g', linewidth=4, markersize=12)
    ax.plot(eye['Eyepiece'].T[0] - 1.0, eye['Eyepiece'].T[1], 'r', linewidth=4, markersize=12)
    ax.plot(eye['Eyepiece'].T[0] - 0.5, eye['Eyepiece'].T[1], 'b', linewidth=4, markersize=12)
    ax.plot(eye['Led'].T[0], eye['Led'].T[1], 'ro', linewidth=2, markersize=12)
    ax.plot(eye['Led'].T[0]+1.0, eye['Led'].T[1], 'rs', linewidth=2, markersize=12)
    ax.plot(eye['pupil'][0], eye['pupil'][1], 'ob', linewidth=2, markersize=12)
    ax.text(eye['pupil'][0], eye['pupil'][1], 'P')
    ax.plot(eye['cornea'][0], eye['cornea'][1], 'ob', linewidth=2, markersize=12)
    ax.text(eye['cornea'][0], eye['cornea'][1], 'C')
    ax.plot(eye['EBC'][0], eye['EBC'][1], 'ob', linewidth=2, markersize=12)
    ax.text(eye['EBC'][0], eye['EBC'][1], 'EBC')
    ax.plot(eye['Cam'][0], eye['Cam'][1], '>m', linewidth=2, markersize=12)
    ax.plot(eye['Cam'][0]+1.0, eye['Cam'][1], 'sm', linewidth=2, markersize=12)
    plt.text(eye['Cam'][0], eye['Cam'][1], 'Cam')
    for i in range(np.shape(eye['glint_beam'])[1]):
        ax.plot(eye['glint_beam'][:, i, 0], eye['glint_beam'][:, i, 1], ':r', linewidth=1, markersize=12)
    ax.axis('equal')
    ax.grid('on', color='gray', linewidth=0.5)
    # plt.show()
    fig.canvas.draw_idle()
    # return ax

    # axcolor = 'lightgoldenrodyellow'
    # axgaze = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    #
    # sgaze = Slider(axgaze, 'Gaze', 0.1, 30.0, valinit=0, valstep=0.5)
    #
    # def update(val):
    #     eye = calcEye(sgaze.val)
    #     fig.canvas.draw_idle()

    # ax.margins(x=0)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(13,6))
    plt.subplots_adjust(left=0.2)
    # gaze = np.array([7., 0.])  # [Deg]
    # eye_2 = calcEye(gaze=gaze_1)
    # ax_2 = plotEye(eye=eye_2)


    axcolor = 'lightgoldenrodyellow'
    axgaze = plt.axes([0.1, 0.1, 0.03, 0.8], facecolor=axcolor)
    sgaze = Slider(axgaze, 'Gaze', -25., 25., valinit=0., valstep=0.5, orientation='vertical')

    def update(val=5.):
        gaze_val = sgaze.val
        eye = calcEye(gaze=gaze_val)
        plotEye(eye=eye)
        # fig.canvas.draw_idle()

    update()
    sgaze.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        ax.clear()
        update()


    button.on_clicked(reset)

    plt.show()
    # print('EBC:', eye_['EBC'], '\nLed: ', eye_['Led'], '\nGlint: ', eye_['glint'][0:8], '\nCcontou: ',
    #       eye_['Ccontour'][0:8], '\nEyepiece: ', eye_['Eyepiece'], '\nlightfieldY: ', eye_['lightfieldY'])
