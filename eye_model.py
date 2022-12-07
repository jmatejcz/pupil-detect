from unprojection import unproject_eye_circle
from utils import calc_intersection


class EyeModeling:
    """https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting"""

    def __init__(self, focal_len, pupil_radius) -> None:
        self.initial_pupil_radius = pupil_radius  # pixels
        self.camera_vertex = [0, 0, -focal_len]  # ???

        self.disc_normals = []
        self.disc_centers = []

    def two_circle_unprojection(self, ellipse):
        """
        notatki.txt -> two circle unprojection
        """
        (
            pupil_normal_pos,
            pupil_normal_neg,
            pupil_centre_pos,
            pupil_centre_neg,
        ) = unproject_eye_circle(
            camera_vertex=self.camera_vertex,
            ellipse=ellipse,
            radius=self.initial_pupil_radius,
        )

        self.disc_normals.append((pupil_normal_pos, pupil_normal_neg))
        self.disc_centers.append((pupil_centre_pos, pupil_centre_neg))

    def sphere_centre_estimate(self):

        # reduction to 2D, and taking pos normal vector
        # TODO można brac oba vektory skoro sa rownoległe ?
        normal_vectors_2D = [vectors[0][0:2] for vectors in self.disc_normals]
        disc_centers_2D = [centers[0][0:2] for centers in self.disc_centers]
        self.estimated_eye_center = calc_intersection(
            normal_vectors_2D, disc_centers_2D
        )

    def sphere_radius_estimate(self):
        