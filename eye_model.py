from unprojection import unproject_eye_circle


class EyeModeling:
    """https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting"""

    def __init__(self, ellipse, focal_len) -> None:
        self.initial_pupil_radius = 2  # mm
        self.ellipse = ellipse
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
            ellipse=self.ellipse,
            radius=self.initial_pupil_radius,
        )

        self.disc_normals.append((pupil_normal_pos, pupil_normal_neg))
        self.disc_centers.append((pupil_centre_pos, pupil_centre_neg))

    def sphere_centre_estimate(self):

        # reduction to 2D, and taking pos normal vector
        normal_vectors_2D = [vectors[0][:, 0:2] for vectors in self.disc_normals]
        disc_centers_2D = [centers[0][:, 0:2] for centers in self.disc_centers]
