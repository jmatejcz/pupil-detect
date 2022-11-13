import numpy as np


def get_general_equation_ellipse_coefficient(x, y, a, b, rot_angle):
    """https://en.wikipedia.org/wiki/Ellipse#General_ellipse

    :param x: x coord, of pupil centre
    :type x: _type_
    :param y: y coord, of pupil centre
    :type y: _type_
    :param a: minor axis of pupil
    :type a: _type_
    :param b: minor axis of pupil
    :type b: _type_
    :param rot_angle: rotation angle of ellipse
    :type rot_angle: _type_
    :return: _description_
    :rtype: _type_
    """
    A = (a**2) * (np.sin(rot_angle) ** 2) + (b**2) * (np.cos(rot_angle) ** 2)
    B = 2 * A((b**2) - (a**2)) * np.sin(rot_angle) * np.cos(rot_angle)
    C = (a**2) * (np.cos(rot_angle) ** 2) + (b**2) * (np.sin(rot_angle) ** 2)
    D = -2 * A * x - B * y
    E = -B * x - 2 * C * y
    F = A * (x**2) + B * x * y + C * (y**2) - (a**2) * (b**2)

    return (A, B, C, D, E, F)

def unproject_eye(pupil_pred):
    
    



