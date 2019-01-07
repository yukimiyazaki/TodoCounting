

black_id_list = (
    # From MismatchedTrainImages.txt
    3,  # Region mismatch
    # 7,     # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    9,  # Region mismatch
    21,  # Region mismatch
    30,  # Exposure mismatch -- not fixable
    34,  # Exposure mismatch -- not fixable
    71,  # Region mismatch
    81,  # Region mismatch
    89,  # Region mismatch
    97,  # Region mismatch
    151,  # Region mismatch
    184,  # Exposure mismatch -- almost fixable
    # 215,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    234,  # Region mismatch
    242,  # Region mismatch
    268,  # Region mismatch
    290,  # Region mismatch
    311,  # Region mismatch
    # 331,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 344,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    380,  # Exposure mismatch -- not fixable
    384,  # Region mismatch
    # 406,   # Exposure mismatch -- fixed by find_coords()
    # 421,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 469,   # Exposure mismatch -- fixed by find_coords()
    # 475,   # Exposure mismatch -- fixed by find_coords()
    490,  # Region mismatch
    499,  # Region mismatch
    507,  # Region mismatch
    # 530,   # TrainDotted rotated. Hot patch in load_dotted_image()
    531,  # Exposure mismatch -- not fixable
    # 605,   # In MismatchedTrainImages, but appears to be O.K.
    # 607,   # Missing annotations on 2 adult males, added to missing_coords
    614,  # Exposure mismatch -- not fixable
    621,  # Exposure mismatch -- not fixable
    # 638,   # TrainDotted rotated. Hot patch in load_dotted_image()
    # 644,   # Exposure mismatch, but not enough to cause problems
    687,  # Region mismatch
    712,  # Exposure mismatch -- not fixable
    721,  # Region mismatch
    767,  # Region mismatch
    779,  # Region mismatch
    # 781,   # Exposure mismatch -- fixed by find_coords()
    # 794,   # Exposure mismatch -- fixed by find_coords()
    800,  # Region mismatch
    811,  # Region mismatch
    839,  # Region mismatch
    840,  # Exposure mismatch -- not fixable
    869,  # Region mismatch
    # 882,   # Exposure mismatch -- fixed by find_coords()
    # 901,   # Train image has (different) mask already, but not actually a problem
    903,  # Region mismatch
    905,  # Region mismatch
    909,  # Region mismatch
    913,  # Exposure mismatch -- not fixable
    927,  # Region mismatch
    946,  # Exposure mismatch -- not fixable

    # Additional anomalies
    129,  # Raft of marked juveniles in water (middle top). But another
    # large group bottom middle are not marked
    200,  # lots of pups marked as adult males
    235,  # None of the 35 adult males have been labelled
    857,  # Missing annotations on all sea lions (Kudos: @depthfirstsearch)
    941,  # 5 adult males not marked
)

# 回転を適用済みです。
rotated_id_list = (
    # From MismatchedTrainImages.txt
    # 7,     # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 215,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 331,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 344,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 421,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 530,   # TrainDotted rotated. Hot patch in load_dotted_image()
    # 638,   # TrainDotted rotated. Hot patch in load_dotted_image()
)