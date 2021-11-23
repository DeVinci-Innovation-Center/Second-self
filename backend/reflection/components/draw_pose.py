from cv2 import cv2


def lerp(P1, P2, f):
    """Simple Linear Interpolation of two values"""
    return P1 + (P2 - P1) * f


class DrawPose:
    """Use Body parametters to draw the body on a provided image"""

    def __init__(self, links, resolution, offset, dimension):

        self.body_junctions = links
        self.resolution = resolution
        self.offset = offset
        self.dimension = dimension

        self.color = (255, 255, 255)
        self.thickness = 5

        self.body_pose = []
        self.body_pose_t = []

        self.show_head = False
        self.show_wrist = True

    def draw(self, image, data):
        """Draws the body on an opencv image"""

        self.body_pose = data["body_pose"]

        if len(self.body_pose) == 0:
            return image

        for i, pose in enumerate(self.body_pose):
            if self.body_pose[2:4] != [-1, -1]:
                if len(self.body_pose_t) == len(self.body_pose):
                    newx = self.resolution[0] * (pose[0] - self.offset[0]) / self.dimension[0]
                    newy = self.resolution[1] * (pose[1] - self.offset[1]) / self.dimension[1]
                    if newy > 0:
                        x = lerp(self.body_pose_t[i][0], newx, 0.8)
                        y = lerp(self.body_pose_t[i][1], newy, 0.8)
                    else:
                        x = lerp(self.body_pose_t[i][0], newx, 0.01)
                        y = lerp(self.body_pose_t[i][1], newy, 0.01)

                    self.body_pose_t[i] = [int(x), int(y)]
                else:
                    x = self.resolution[0] * (pose[0] - self.offset[0]) / self.dimension[0]
                    y = self.resolution[1] * (pose[1] - self.offset[1]) / self.dimension[1]
                    self.body_pose_t.append([int(x), int(y)])

        for parts in self.body_junctions:
            for pair in parts:
                image = cv2.line(
                    image,
                    self.body_pose_t[pair[0]],
                    self.body_pose_t[pair[1]],
                    self.color,
                    self.thickness,
                )
        return image
