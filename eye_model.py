from unprojection import unproject_eye_circle
from utils import calc_intersection
import numpy as np


class EyeModeling:
    """
    https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting
    Sometimes I will use variable names according to the paper
    n - pupil disc normal vector
    c - eye sphere center
    p - pupil center
    """

    def __init__(self, focal_len, pupil_radius, image_shape, inital_z) -> None:
        self.initial_pupil_radius = pupil_radius  # pixels
        self.camera_vertex = [0, 0, -focal_len]  # ???
        self.image_shape = image_shape

        self.disc_normals = []
        self.disc_centers = []

        self.initial_z_of_eye_center = inital_z

    def two_circle_unprojection(self, ellipse):
        """
        notatki.txt -> two circle unprojection
        """
        x, y = ellipse[0]
        a, b = ellipse[1]
        rot = ellipse[2]
        # need to switch for coordinates where camera is in the middle
        # so (0,0) point is in the middle
        x -= self.image_shape[1] / 2
        y -= self.image_shape[0] / 2

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
        # print(pupil_normal_pos, pupil_normal_neg)
        # print(pupil_centre_pos, pupil_centre_neg)

        self.disc_normals.append((pupil_normal_pos, pupil_normal_neg))
        self.disc_centers.append((pupil_centre_pos, pupil_centre_neg))

    def sphere_centre_estimate(self):

        # reduction to 2D, and taking pos normal vector
        # TODO można brac oba vektory skoro sa rownoległe ?
        normal_vectors_2D = [vectors[0][0:2] for vectors in self.disc_normals]
        disc_centers_2D = [centers[0][0:2] for centers in self.disc_centers]
        self.estimated_eye_center_2D = calc_intersection(
            normal_vectors_2D, disc_centers_2D
        )

    def sphere_radius_estimate(self):
        """
        Now we can discard 1 of 2 solutions for normal vector and center
        this has to be true : n * (c-p) > 0, also in 2D
        Then we can calculate intersection between (camera_vertex, p) 
        and (c,p) to estimate sphere radius
        """
        self.filtered_disc_normals = []
        self.filtered_disc_centers = []
        for i in range(len(self.disc_normals)):

            n = self.disc_normals[i][0][0:2]
            c = self.estimated_eye_center_2D
            p = self.disc_centers[i][0][0:2]
            result = n*(c-p)
            if result[0] > 0 and result[1] > 0:
                self.filtered_disc_normals.append(self.disc_normals[i][0])
                self.filtered_disc_centers.append(self.disc_centers[i][0])

            n = self.disc_normals[i][1][0:2]
            c = self.estimated_eye_center_2D
            p = self.disc_centers[i][1][0:2]
            result = n*(c-p)
            if result[0] > 0 and result[1] > 0:
                self.filtered_disc_normals.append(self.disc_normals[i][1])
                self.filtered_disc_centers.append(self.disc_centers[i][1])

        # now we can calculate intersection
        # we take mean from all intersections
        # but first, we need a 3D coords of eye center, so we have to fix some value of z
        # and scale it with camera vertex
        self.estimated_eye_center_3D = self.estimated_eye_center_2D * \
            self.initial_z_of_eye_center/self.camera_vertex[2]  # focal length

        radiuses = []
        for i in range(len(self.filtered_disc_normals)):
            # TODO czy na pewno vektor to p?
            vectors = [self.filtered_disc_normals[i], self.filtered_disc_centers[i] /
                       np.linalg.norm(self.filtered_disc_centers[i])]
            # [0, 0, 0] is origin of camera so disc center can albo be vector to itself
            centers = [self.estimated_eye_center_3D,
                       self.filtered_disc_centers[i]]
            # p = calc_intersection(self.filtered_disc_normals, self.filtered_disc_centers)

            radiuses.append(self.estimated_eye_center_2D)
        return np.mean(radiuses)
