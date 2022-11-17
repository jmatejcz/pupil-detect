import numpy as np

# TODO Z ZRP STRONKA GURU OF THE WEEK ZADANIA


def get_general_equation_ellipse_coefficients(x, y, a, b, rot_angle):
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


def get_cone_coefficients(
    alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime
):
    """Get general cone equation coefficients from camera vertex and base ellipse equation

    (alpha, beta, gamma) = camera vertex
    """
    gamma_square = np.power(gamma, 2)
    a = gamma_square * a_prime
    b = gamma_square * b_prime
    c = (
        a_prime * np.power(alpha, 2)
        + 2 * h_prime * alpha * beta
        + b_prime * np.power(beta, 2)
        + 2 * g_prime * alpha
        + 2 * f_prime * beta
        + d_prime
    )
    d = gamma_square * d_prime
    f = -gamma * (b_prime * beta + h_prime * alpha + f_prime)
    g = -gamma * (h_prime * beta + a_prime * alpha + g_prime)
    h = gamma_square * h_prime
    u = gamma_square * g_prime
    v = gamma_square * f_prime
    w = -gamma * (f_prime * beta + g_prime * alpha + d_prime)
    return a, b, c, d, f, g, h, u, v, w


def get_plane_coefiicient(_lambda1, _lambda2, _lambda3):
    """
    there is 3 cases as below
    """
    if _lambda1 < _lambda2:
        l = 0
        m_pos = np.sqrt((_lambda2 - _lambda1) / (_lambda2 - _lambda3))
        m_neg = -m_pos
        n = np.sqrt((_lambda1 - _lambda3) / (_lambda2 - _lambda3))
        return [l, l], [m_pos, m_neg], [n, n]
    elif _lambda1 > _lambda2:
        l_pos = np.sqrt((_lambda1 - _lambda2) / (_lambda1 - _lambda3))
        l_neg = -l_pos
        n = np.sqrt((_lambda2 - _lambda3) / (_lambda1 - _lambda3))
        m = 0
        return [l_pos, l_neg], [m, m], [n, n]
    elif _lambda1 == _lambda2:
        n = 1
        m = 0
        l = 0
        return [l, l], [m, m], [n, n]
    else:
        return None, None, None


def gen_rotation_transofrm_coefficents(_lambda, a, b, g, f, h):

    t1 = (b - _lambda) * g - f * h
    t2 = (a - _lambda) * f - g * h
    t3 = -(a - _lambda) * (t1 / t2) / g - (h / g)
    m = 1 / (np.sqrt(1 + np.power((t1 / t2), 2) + np.power(t3, 2)))
    l = (t1 / t2) * m
    n = t3 * m

    return l, m, n


def unproject_eye(camera_vertex, ellipse, radius=None):
    """_based on Three-dimensional location estimation of circular features for machine vision by
    R. Safaee-Rad; I. Tchoukanov; K.C. Smith; B. Benhabib -> https://ieeexplore.ieee.org/document/163786

    :param camera_vertex: center of camera lens
    :type camera_vertex: tuple - camera vertex coords in 3D with respect to
    :param ellipse: (x, y, a, b, rot_angle)
        where x,y -> ellipse center, a,b -> axis, rot_angle -> rotation angle

    """
    A, B, C, D, E, F = get_general_equation_ellipse_coefficients(*ellipse)

    # step 1)
    # (1) in algorithm paper
    a_prime = A
    h_prime = B / 2
    b_prime = C
    g_prime = D / 2
    f_prime = E / 2
    d_prime = F

    # step 2)
    # The coefficients of a general cone equation
    # (3) in algorithm paper
    a, b, c, d, f, g, h, u, v, w = get_cone_coefficients(
        *camera_vertex, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime
    )

    # step 3)
    # Reduction of the equation of a cone
    # (10) in algorithm paper
    # coef1, coef2, coef3, coef4 = get__lamba_coefficients(a, b, c, f, g, h)
    lambda_coefficient_1 = 1
    lambda_coefficient_2 = -(a + b + c)
    lambda_coefficient_3 = (
        b * c + c * a + a * b - np.power(f, 2) - np.power(g, 2) - np.power(h, 2)
    )
    lambda_coefficient_4 = -(
        a * b * c
        + 2 * f * g * h
        - a * np.power(f, 2)
        - b * np.power(g, 2)
        - c * np.power(h, 2)
    )

    # now we find the roots of polynominal with above coefficient
    lambda1, lambda2, lambda3 = np.roots(
        [
            lambda_coefficient_1,
            lambda_coefficient_2,
            lambda_coefficient_3,
            lambda_coefficient_4,
        ]
    )

    # step 4)
    # The coefficients of the equation of the circular-feature plane
    # (27) in algorithm paper
    # 3 cases occur
    l, m, n = get_plane_coefiicient(lambda1, lambda2, lambda3)

    # find elements of a rotational transformation
    # (8) , (12) in algorithm paper
    # one rotational coefficietns per lambda
    l1, m1, n1 = gen_rotation_transofrm_coefficents(lambda1, a, b, g, f, h)
    l2, m2, n2 = gen_rotation_transofrm_coefficents(lambda2, a, b, g, f, h)
    l3, m3, n3 = gen_rotation_transofrm_coefficents(lambda3, a, b, g, f, h)

    # rotational transform matrix
    T1 = np.array([l1, l2, l3, 0, m1, m2, m3, 0, n1, n2, n3, 0, 0, 0, 0, 1]).reshape(
        4, 4
    )

    # step 5)
    #

    # get plane's normal vector
    normal_vec_pos = np.array([l[0], m[0], n[0], 1]).reshape(4, 1)
    normal_vec_neg = np.array([l[1], m[1], n[1], 1]).reshape(4, 1)

    # applying the transformation to the surface's normal vector
    # it gives as normal vector with respect to the camera frame
    normal_cam_pos = np.dot(T1, normal_vec_pos)
    normal_cam_neg = np.dot(T1, normal_vec_neg)
