# MotionStage Stage 19B
# Build and animate a retarget-ready humanoid bridge rig in Blender.
#
# Run INSIDE Blender after Stage 19A:
#   1. Keep the Stage 19A scene open
#   2. Go to Scripting
#   3. Open this file
#   4. Press Run Script
#
# The script reads:
#   ~/Desktop/MotionStage/data/animation/test_performance_01/animation_package.json
#   ~/Desktop/MotionStage/backend/motionstage/animation/motionstage_retarget_map.json
#
# It creates:
#   MotionStage_RetargetRig
#
# The new rig is placed beside the Stage 19A source rig for visual comparison.

import bpy
import json
from pathlib import Path
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "MotionStage"
)

PACKAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "animation"
    / "test_performance_01"
    / "animation_package.json"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "backend"
    / "motionstage"
    / "animation"
    / "motionstage_retarget_map.json"
)

COLLECTION_NAME = "MotionStage_19B"
ARMATURE_NAME = "MotionStage_RetargetRig"

# Put the bridge rig beside the Stage 19A source rig.
DISPLAY_OFFSET_X = 1.1

# Small visual root bone size.
ROOT_BONE_LENGTH = 0.08

# Synthetic hand extension relative to forearm length.
HAND_EXTENSION_RATIO = 0.28


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required MotionStage file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def as_vector(values):
    return Vector(
        (
            float(values[0]),
            float(values[1]),
            float(values[2]),
        )
    )


def midpoint(a, b):
    return (a + b) * 0.5


def lerp(a, b, t):
    return a + ((b - a) * t)


def safe_direction(head, tail):
    direction = tail - head

    if direction.length < 1e-7:
        direction = Vector(
            (0.0, 0.0, 0.05)
        )

    return direction


def remove_existing_collection(name):
    collection = bpy.data.collections.get(
        name
    )

    if collection is None:
        return

    for obj in list(
        collection.objects
    ):
        bpy.data.objects.remove(
            obj,
            do_unlink=True,
        )

    bpy.data.collections.remove(
        collection
    )


def create_collection(name):
    collection = bpy.data.collections.new(
        name
    )

    bpy.context.scene.collection.children.link(
        collection
    )

    return collection


def build_bridge_points(joints):
    """
    Convert MotionStage Stage 19A joints into the joints needed for a
    more standard humanoid bridge skeleton.
    """

    pelvis = as_vector(
        joints["pelvis"]
    )

    chest = as_vector(
        joints["chest"]
    )

    head = as_vector(
        joints["head"]
    )

    left_shoulder = as_vector(
        joints["left_shoulder"]
    )

    right_shoulder = as_vector(
        joints["right_shoulder"]
    )

    left_elbow = as_vector(
        joints["left_elbow"]
    )

    right_elbow = as_vector(
        joints["right_elbow"]
    )

    left_wrist = as_vector(
        joints["left_wrist"]
    )

    right_wrist = as_vector(
        joints["right_wrist"]
    )

    left_hip = as_vector(
        joints["left_hip"]
    )

    right_hip = as_vector(
        joints["right_hip"]
    )

    left_knee = as_vector(
        joints["left_knee"]
    )

    right_knee = as_vector(
        joints["right_knee"]
    )

    left_ankle = as_vector(
        joints["left_ankle"]
    )

    right_ankle = as_vector(
        joints["right_ankle"]
    )

    left_foot = as_vector(
        joints["left_foot"]
    )

    right_foot = as_vector(
        joints["right_foot"]
    )

    # We have only pelvis -> spine -> chest in the source package.
    # Split that torso span into three bridge segments.
    spine_01_point = lerp(
        pelvis,
        chest,
        0.34,
    )

    spine_02_point = lerp(
        pelvis,
        chest,
        0.67,
    )

    # Synthetic neck joint between upper chest and tracked head/nose.
    neck = lerp(
        chest,
        head,
        0.38,
    )

    # Extend each wrist along the forearm direction to create a short hand bone.
    left_forearm = (
        left_wrist
        - left_elbow
    )

    right_forearm = (
        right_wrist
        - right_elbow
    )

    left_hand = (
        left_wrist
        + left_forearm
        * HAND_EXTENSION_RATIO
    )

    right_hand = (
        right_wrist
        + right_forearm
        * HAND_EXTENSION_RATIO
    )

    return {
        "root_head":
            pelvis,

        "root_tail":
            pelvis
            + Vector(
                (
                    0.0,
                    0.0,
                    ROOT_BONE_LENGTH,
                )
            ),

        "pelvis":
            pelvis,

        "spine_01_point":
            spine_01_point,

        "spine_02_point":
            spine_02_point,

        "chest":
            chest,

        "neck":
            neck,

        "head":
            head,

        "left_shoulder":
            left_shoulder,

        "left_elbow":
            left_elbow,

        "left_wrist":
            left_wrist,

        "left_hand":
            left_hand,

        "right_shoulder":
            right_shoulder,

        "right_elbow":
            right_elbow,

        "right_wrist":
            right_wrist,

        "right_hand":
            right_hand,

        "left_hip":
            left_hip,

        "left_knee":
            left_knee,

        "left_ankle":
            left_ankle,

        "left_foot":
            left_foot,

        "right_hip":
            right_hip,

        "right_knee":
            right_knee,

        "right_ankle":
            right_ankle,

        "right_foot":
            right_foot,
    }


def bridge_bone_geometry(points):
    """
    Head/tail definition for every Stage 19B bridge bone.
    """

    return {
        "root": (
            points["root_head"],
            points["root_tail"],
        ),

        "pelvis": (
            points["pelvis"],
            points["spine_01_point"],
        ),

        "spine_01": (
            points["spine_01_point"],
            points["spine_02_point"],
        ),

        "spine_02": (
            points["spine_02_point"],
            points["chest"],
        ),

        "neck_01": (
            points["chest"],
            points["neck"],
        ),

        "head": (
            points["neck"],
            points["head"],
        ),

        "clavicle_l": (
            points["chest"],
            points["left_shoulder"],
        ),

        "upperarm_l": (
            points["left_shoulder"],
            points["left_elbow"],
        ),

        "lowerarm_l": (
            points["left_elbow"],
            points["left_wrist"],
        ),

        "hand_l": (
            points["left_wrist"],
            points["left_hand"],
        ),

        "clavicle_r": (
            points["chest"],
            points["right_shoulder"],
        ),

        "upperarm_r": (
            points["right_shoulder"],
            points["right_elbow"],
        ),

        "lowerarm_r": (
            points["right_elbow"],
            points["right_wrist"],
        ),

        "hand_r": (
            points["right_wrist"],
            points["right_hand"],
        ),

        "thigh_l": (
            points["left_hip"],
            points["left_knee"],
        ),

        "calf_l": (
            points["left_knee"],
            points["left_ankle"],
        ),

        "foot_l": (
            points["left_ankle"],
            points["left_foot"],
        ),

        "thigh_r": (
            points["right_hip"],
            points["right_knee"],
        ),

        "calf_r": (
            points["right_knee"],
            points["right_ankle"],
        ),

        "foot_r": (
            points["right_ankle"],
            points["right_foot"],
        ),
    }


def desired_bone_matrix(
    head,
    tail,
    rest_length,
):
    direction = safe_direction(
        head,
        tail,
    )

    target_length = (
        direction.length
    )

    rotation = (
        direction
        .normalized()
        .to_track_quat(
            "Y",
            "Z",
        )
        .to_matrix()
        .to_4x4()
    )

    scale_y = (
        target_length
        / rest_length
        if rest_length > 1e-8
        else 1.0
    )

    scale = Matrix.Diagonal(
        (
            1.0,
            scale_y,
            1.0,
            1.0,
        )
    )

    return (
        Matrix.Translation(
            head
        )
        @ rotation
        @ scale
    )


def mapping_hierarchy(mapping):
    hierarchy = mapping.get(
        "bridge_hierarchy",
        []
    )

    if not hierarchy:
        raise ValueError(
            "Retarget mapping has no bridge_hierarchy."
        )

    return hierarchy


def create_armature(
    collection,
    package,
    mapping,
):
    first_frame = package[
        "frames"
    ][0]

    points = build_bridge_points(
        first_frame[
            "joints"
        ]
    )

    geometry = bridge_bone_geometry(
        points
    )

    hierarchy = mapping_hierarchy(
        mapping
    )

    required_names = {
        item["name"]
        for item in hierarchy
    }

    missing_geometry = (
        required_names
        - set(geometry)
    )

    if missing_geometry:
        raise ValueError(
            "Missing bridge geometry for bones: "
            f"{sorted(missing_geometry)}"
        )

    armature_data = bpy.data.armatures.new(
        ARMATURE_NAME
    )

    armature_object = bpy.data.objects.new(
        ARMATURE_NAME,
        armature_data,
    )

    armature_object.location.x = (
        DISPLAY_OFFSET_X
    )

    collection.objects.link(
        armature_object
    )

    bpy.context.view_layer.objects.active = (
        armature_object
    )

    armature_object.select_set(
        True
    )

    bpy.ops.object.mode_set(
        mode="EDIT"
    )

    edit_bones = {}

    for item in hierarchy:
        name = item["name"]

        head, tail = geometry[
            name
        ]

        tail = (
            head
            + safe_direction(
                head,
                tail,
            )
        )

        bone = (
            armature_data
            .edit_bones
            .new(name)
        )

        bone.head = head
        bone.tail = tail

        edit_bones[
            name
        ] = bone

    for item in hierarchy:
        parent_name = item.get(
            "parent"
        )

        if not parent_name:
            continue

        child = edit_bones[
            item["name"]
        ]

        parent = edit_bones[
            parent_name
        ]

        child.parent = parent

        # Preserve exact tracked head positions instead of forcing a connected
        # chain. This is safer for the MediaPipe-derived bridge.
        child.use_connect = False

    bpy.ops.object.mode_set(
        mode="POSE"
    )

    for pose_bone in (
        armature_object
        .pose
        .bones
    ):
        pose_bone.rotation_mode = (
            "QUATERNION"
        )

    bpy.ops.object.mode_set(
        mode="OBJECT"
    )

    return armature_object


def ordered_bone_names(mapping):
    hierarchy = mapping_hierarchy(
        mapping
    )

    parent_lookup = {
        item["name"]:
            item.get("parent")
        for item in hierarchy
    }

    remaining = set(
        parent_lookup
    )

    ordered = []

    while remaining:
        progress = False

        for name in list(
            remaining
        ):
            parent = parent_lookup[
                name
            ]

            if (
                parent is None
                or parent in ordered
                or parent
                not in parent_lookup
            ):
                ordered.append(
                    name
                )

                remaining.remove(
                    name
                )

                progress = True

        if not progress:
            ordered.extend(
                sorted(
                    remaining
                )
            )
            break

    return ordered


def animate_armature(
    package,
    mapping,
    armature_object,
):
    order = ordered_bone_names(
        mapping
    )

    rest_lengths = {
        name:
            max(
                float(
                    armature_object
                    .data
                    .bones[
                        name
                    ]
                    .length
                ),
                1e-8,
            )
        for name in order
    }

    bpy.context.view_layer.objects.active = (
        armature_object
    )

    armature_object.select_set(
        True
    )

    bpy.ops.object.mode_set(
        mode="POSE"
    )

    for blender_frame, source in enumerate(
        package["frames"],
        start=1,
    ):
        bpy.context.scene.frame_set(
            blender_frame
        )

        points = build_bridge_points(
            source[
                "joints"
            ]
        )

        geometry = bridge_bone_geometry(
            points
        )

        for name in order:
            head, tail = geometry[
                name
            ]

            pose_bone = (
                armature_object
                .pose
                .bones[
                    name
                ]
            )

            pose_bone.matrix = (
                desired_bone_matrix(
                    head=head,
                    tail=tail,
                    rest_length=
                        rest_lengths[
                            name
                        ],
                )
            )

            pose_bone.keyframe_insert(
                data_path="location",
                frame=blender_frame,
            )

            pose_bone.keyframe_insert(
                data_path=
                    "rotation_quaternion",
                frame=blender_frame,
            )

            pose_bone.keyframe_insert(
                data_path="scale",
                frame=blender_frame,
            )

        bpy.context.view_layer.update()

    bpy.ops.object.mode_set(
        mode="OBJECT"
    )


def configure_display(
    armature_object,
):
    armature_object.show_in_front = (
        True
    )

    armature_object.data.display_type = (
        "OCTAHEDRAL"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

package = load_json(
    PACKAGE_PATH
)

mapping = load_json(
    MAPPING_PATH
)

if (
    mapping.get("schema")
    != "motionstage.retarget_map.v1"
):
    raise ValueError(
        "Unexpected MotionStage retarget mapping schema: "
        f"{mapping.get('schema')}"
    )

remove_existing_collection(
    COLLECTION_NAME
)

collection = create_collection(
    COLLECTION_NAME
)

retarget_rig = create_armature(
    collection,
    package,
    mapping,
)

configure_display(
    retarget_rig
)

animate_armature(
    package,
    mapping,
    retarget_rig,
)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(
    package["frames"]
)
bpy.context.scene.frame_set(1)

# Select the new rig at the end so Frame Selected works immediately.
bpy.ops.object.select_all(
    action="DESELECT"
)

retarget_rig.select_set(
    True
)

bpy.context.view_layer.objects.active = (
    retarget_rig
)

print()
print("=" * 72)
print(
    "MotionStage Stage 19B - "
    "Retarget Bridge Rig"
)
print("=" * 72)
print(
    "Performance        :",
    package[
        "performance_name"
    ],
)
print(
    "Source rig         :",
    mapping.get(
        "source_rig"
    ),
)
print(
    "Retarget rig       :",
    ARMATURE_NAME,
)
print(
    "Frames             :",
    len(
        package[
            "frames"
        ]
    ),
)
print(
    "Bridge bones       :",
    len(
        mapping[
            "bridge_hierarchy"
        ]
    ),
)
print(
    "Synthetic bones    :",
    ", ".join(
        item["name"]
        for item
        in mapping[
            "bridge_hierarchy"
        ]
        if item.get(
            "synthetic"
        )
    ),
)
print(
    "Display offset X   :",
    DISPLAY_OFFSET_X,
)
print()
print(
    "The Stage 19B rig has been placed beside "
    "the Stage 19A source rig."
)
print(
    "Press Space to play and compare both rigs."
)
print("=" * 72)
