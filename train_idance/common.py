CATEGORY_TO_ID = {
    "ballet": 0,
    "kpop": 1,
    "latin": 2,
    "classical-chinese": 3,
    "modern": 4,
}

CATEGORIES = list(CATEGORY_TO_ID.keys())

# https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
# UNUSED_KEYPOINTS = [17, 18, 20, 21, 23, 24]
UNUSED_KEYPOINTS = []
KEYPOINTS = [i for i in range(25) if i not in UNUSED_KEYPOINTS]

def get_vid(filename: str) -> str:
    filename = filename.split('.')[0]
    cat, idx = filename.split('_')
    vid = f"{cat}_{int(idx)}"
    return vid
