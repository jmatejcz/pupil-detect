from unprojection import unproject_eye_circle
from utils import (
    calc_intersection,
    calc_sphere_line_intersection,
    projection,
    calc_angle_between_2D_vectors,
)
from exceptions import NoIntersection
import numpy as np


class EyeModeling:
    """
    https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting
    variable names matching in paper:
    n - pupil disc normal vector
    c - eye sphere center
    p - pupil center
    """

    def __init__(self, focal_len, pupil_radius, image_shape, inital_z) -> None:
        """
        :param focal_len: focal length of camera in pixels
        :param pupil_radius: in pixels
        :param inital_z: initial middle of the eye 'z' coord
        """
        self.initial_pupil_radius = pupil_radius  # pixels
        self.focal_len = focal_len  # in pixels
        self.camera_vertex = [0, 0, -focal_len]  # ???
        self.image_shape = image_shape

        self.disc_normals = []
        self.disc_centers = []
        self.ellipses = []
        # self.ellipse_centers = []

        self.initial_z_of_eye_center = inital_z

    def two_circle_unprojection(self, ellipse) -> tuple:
        """
        README.txt -> two circle unprojection
        Returns:
            disc_normals -> ([[x], [y], [z]], [[x], [y], [z]])
            disc_centers -> ([[x], [y], [z]], [[x], [y], [z]])
            shape -> (3, 1)
        """
        x, y = ellipse[0]
        a, b = ellipse[1]
        rot = ellipse[2]

        # need to switch for coordinates with respect to camera in the middle
        # so (0,0) point is in the middle
        x -= self.image_shape[1] / 2
        y -= self.image_shape[0] / 2
        self.ellipses.append(np.array([x, y, *ellipse[1], ellipse[2]]))
        if x == 0.0:
            x = 1.0
        if y == 0.0:
            y = 1.0

        (
            pupil_normal_pos,
            pupil_normal_neg,
            pupil_centre_pos,
            pupil_centre_neg,
        ) = unproject_eye_circle(
            camera_vertex=self.camera_vertex,
            ellipse=[x, y, a, b, rot],
            radius=self.initial_pupil_radius,
        )

        unprojected_vectors = (pupil_normal_pos, pupil_normal_neg)
        unprojected_centers = (pupil_centre_pos, pupil_centre_neg)

        self.disc_normals.append(unprojected_vectors)
        self.disc_centers.append(unprojected_centers)

        return (unprojected_vectors, unprojected_centers)

    def sphere_centre_estimate(self):
        """
        README.txt -> SPHERE CENTRE ESTIMATE
        intersection of the normal vectors is a sphere center
        Returns: sphere center 2D ->[[x], [y]]
        shape -> (2,1)

        """
        normal_vectors_2D = []
        disc_centers_2D = []
        # filter ellipses to take these with bigger axis length difference
        for i, ellipse in enumerate(self.ellipses):
            angle = calc_angle_between_2D_vectors(
                vec1=self.disc_normals[i][0][0:2], vec2=self.disc_normals[i][1][0:2]
            )
            # if abs(angle) < 40 or abs(angle) > 165:

            # reduction to 2D, and taking pos normal vector
            normal_vectors_2D.append(self.disc_normals[i][0][0:2])
            normal_vectors_2D.append(self.disc_normals[i][1][0:2])
            # predicted centers 2D
            # disc_centers_2D.append(self.disc_centers[i][0][0:2])
            # disc_centers_2D.append(self.disc_centers[i][1][0:2])
            disc_centers_2D.append(self.ellipses[i][0:2])
            disc_centers_2D.append(self.ellipses[i][0:2])

        self.estimated_eye_center_2D = calc_intersection(
            normal_vectors_2D, disc_centers_2D
        )
        # we don't have z coord z of eye center so for now we fix it with some initial value
        # and scale x, y with regard to camera
        self.estimated_eye_center_3D = (
            self.estimated_eye_center_2D * self.initial_z_of_eye_center / self.focal_len
        )
        self.estimated_eye_center_3D = np.append(
            self.estimated_eye_center_3D, self.initial_z_of_eye_center
        ).reshape(3, 1)

        return self.estimated_eye_center_2D, self.estimated_eye_center_3D

    def filter_vectors_towards_center(
        self, disc_normals=None, disc_centers=None
    ) -> tuple:
        """
        Returns only vectors with their disc centers pointing towards sphere center
        filtered_disc_normals -> [[x], [y], [z]]
        filtered_disc_centers -> [[x], [y], [z]]
        shape -> (3, 1)
        """
        filtered_disc_normals = []
        filtered_disc_centers = []

        for i in range(len(disc_normals)):
            projected_centre = projection(self.estimated_eye_center_3D, self.focal_len)
            projected_normal = projection(
                disc_normals[i][0] + disc_centers[i][0], self.focal_len
            )
            projected_pos = projection(disc_centers[i][0], self.focal_len)
            if (
                np.dot(projected_normal.T, projected_pos - projected_centre).ravel()[0]
                > 0
            ):
                filtered_disc_normals.append(disc_normals[i][0])
                filtered_disc_centers.append(disc_centers[i][0])
            else:
                filtered_disc_normals.append(disc_normals[i][1])
                filtered_disc_centers.append(disc_centers[i][1])

        return filtered_disc_normals, filtered_disc_centers

    def sphere_radius_estimate(self):
        """
        README.txt -> SPHERE RADIUS ESTIMATE
        Now we can discard 1 of 2 solutions for normal vector and center
        this has to be true : n * (c-p) > 0, also in 2D
        Then we can calculate intersection between (camera_vertex, p)
        and (c,p) to estimate sphere radius
        """
        (
            self.filtered_disc_normals,
            self.filtered_disc_centers,
        ) = self.filter_vectors_towards_center(self.disc_normals, self.disc_centers)

        # now we can calculate intersection
        # we take mean from all intersections
        radiuses = []
        for i in range(len(self.filtered_disc_normals)):
            vectors = [
                self.filtered_disc_normals[i],
                self.filtered_disc_centers[i]
                / np.linalg.norm(self.filtered_disc_centers[i]),
            ]
            # [0, 0, 0] is origin so disc center can also be vector to itself
            # centers = [self.estimated_eye_center_3D, self.filtered_disc_centers[i]]
            centers = [self.estimated_eye_center_3D, np.array([0, 0, 0])]
            p = calc_intersection(vectors, centers)

            radiuses.append(np.linalg.norm(p - self.estimated_eye_center_3D))

        # self.estimated_sphere_radius = np.mean(radiuses)
        self.estimated_sphere_radius = 10.5 * 166.95652173913044
        return np.mean(radiuses)

    def consistent_pupil_estimate(self, pupil_pos):
        """
        README.txt -> CONSISTENT PUPIL ESTIMATE
        Depending on estimated eye center and radius,
        calculate new pupil circle (p', n' ,r')
        tangent to the eye sphere where
        p' = sp
        p'=c +Rn'
        r'/z' = r/z
        """
        # s is found as intersection of sphere with (camera, pupil) line
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        u = pupil_pos / np.linalg.norm(pupil_pos)  # unit direction vector
        o = np.array([0, 0, 0]).reshape(3, 1)  # origin of  line
        c = self.estimated_eye_center_3D
        sphere_r = self.estimated_sphere_radius
        intersection = calc_sphere_line_intersection(
            u=u,
            o=o,
            c=c,
            r=sphere_r,
        )
        # we take nearest of 2 points
        if intersection:
            s = min(intersection)
            p_prime = s * u
            n_prime = (p_prime - c) / sphere_r
            n_prime = n_prime / np.linalg.norm(n_prime)
            # take the z and z_prime from p and p_prime as a z coord
            r_prime = (self.initial_pupil_radius / pupil_pos[2]) * p_prime[2]

            return p_prime, n_prime, r_prime
