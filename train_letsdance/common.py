CATEGORIES = {
    "ballet",
    "break",
    "cha",
    "flamenco",
    "foxtrot",
    "jive",
    "latin",
    "pasodoble",
    "quickstep",
    "rumba",
    "samba",
    "square",
    "swing",
    "tango",
    "tap",
    "waltz",
}

CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

KEYPOINTS = [
    # TODO: use what keypoint to represent head
    # "nose",
    "left_eye",
    "right_eye",
    # "left_ear",
    # "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
