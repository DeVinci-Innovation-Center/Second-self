from typing import List

BODY_KEYPOINTS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

BODY_LINKS = [
    [[0, 1], [0, 4], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 8]],
    [[9, 10]],
    [
        [11, 12],
        [11, 13],
        [11, 23],
        [12, 14],
        [12, 24],
        [13, 15],
        [14, 16],
        [15, 17],
        [15, 19],
        [15, 21],
        [16, 18],
        [16, 20],
        [16, 22],
        [17, 19],
        [18, 20],
        [23, 24],
        [23, 25],
        [24, 26],
        [25, 27],
        [26, 28],
        [27, 29],
        [27, 31],
        [28, 30],
        [28, 32],
    ],
]


def normalize_data(data: List[List], width, height) -> List[List]:
    """Normalize the data to fit the hand sign inout data"""
    return [[x / width, y / height] for x, y, *_ in data]
