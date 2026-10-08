# MotionStage Stage 19E.4 — ABSOLUTE BODY-SPACE RETARGET (V7.3 FOOT-AWARE)
#
# Built from the Stage 19E diagnostic data.
#
# Key fix:
# Earlier versions treated MotionStage frame 1 as a per-bone "neutral" pose.
# It is NOT neutral: it is already an actual performance pose.
# That erased the absolute arm/leg pose and made the Mixamo character move
# relative to a T-pose instead of matching the captured body configuration.
#
# V7 does NOT use frame 1 as the neutral pose for individual limbs.
#
# Instead:
#   1. Build a semantic body coordinate frame from hips + torso.
#   2. Express each MotionStage bone direction in that body frame.
#   3. Transfer that ABSOLUTE body-relative direction to Mixamo.
#   4. Preserve Mixamo's own rest roll with a swing-only solve.
#   5. Use frame 1 only to align the two BODY coordinate systems globally.
#
# No root translation or bone scale is baked.
#
# Run INSIDE Blender with:
#   - MotionStage_RetargetRig present
#   - clean Mixamo T-pose character imported
#
# Expected result:
#   The Mixamo character should reproduce the actual captured pose rather than
#   treating the first captured pose as a T-pose reference.

import bpy
from math import radians
from mathutils import Matrix, Quaternion, Vector


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

SOURCE_ARMATURE_NAME = "MotionStage_RetargetRig"
TARGET_REQUIRED_BONE = "mixamorig:Hips"

ACTION_NAME = "MotionStage_Retargeted_Mixamo_V7_3_FootAware"

HIDE_MOTIONSTAGE_VALIDATION = True

# Mixamo's visual forward direction is opposite to the semantic body-forward
# convention used by the MotionStage bridge. Rotate the entire target body
# coordinate frame by 180 degrees around its LOCAL vertical axis.
FACING_YAW_DEGREES = 180.0

# Small constant posture calibration.
#
# In V7.1 the retarget is structurally correct, but the Mixamo character shows
# a consistent backward lean. Because this is a global calibration offset
# rather than a per-limb failure, correct it once in the target BODY frame.
#
# BODY local axes are:
#   +X = anatomical right
#   +Y = anatomical forward
#   +Z = anatomical up
#
# Negative X rotation tips the body slightly forward.
BODY_PITCH_DEGREES = -6.0


# Direct anatomical segments.
#
# Stage 19E.4 change:
# MotionStage foot_l / foot_r are REAL tracked bridge bones (not synthetic),
# so the Mixamo feet are now driven explicitly instead of merely inheriting
# calf rotation. Hands, shoulders/clavicles, toes and fingers still inherit.
BONE_MAP = {
    "spine_01": "mixamorig:Spine",
    "spine_02": "mixamorig:Spine1",

    "neck_01": "mixamorig:Neck",
    "head": "mixamorig:Head",

    "upperarm_l": "mixamorig:LeftArm",
    "lowerarm_l": "mixamorig:LeftForeArm",

    "upperarm_r": "mixamorig:RightArm",
    "lowerarm_r": "mixamorig:RightForeArm",

    "thigh_l": "mixamorig:LeftUpLeg",
    "calf_l": "mixamorig:LeftLeg",
    "foot_l": "mixamorig:LeftFoot",

    "thigh_r": "mixamorig:RightUpLeg",
    "calf_r": "mixamorig:RightLeg",
    "foot_r": "mixamorig:RightFoot",
}


TARGET_ORDER = [
    "mixamorig:Spine",
    "mixamorig:Spine1",

    "mixamorig:Neck",
    "mixamorig:Head",

    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",

    "mixamorig:RightArm",
    "mixamorig:RightForeArm",

    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",

    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
]


TARGET_TO_SOURCE = {
    target_name: source_name
    for source_name, target_name in BONE_MAP.items()
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def find_mixamo_armature():
    candidates = []

    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue

        if obj.name == SOURCE_ARMATURE_NAME:
            continue

        if TARGET_REQUIRED_BONE in obj.data.bones:
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(
            "No Mixamo armature found. Expected bone "
            f"'{TARGET_REQUIRED_BONE}'."
        )

    candidates.sort(
        key=lambda obj: (
            0 if obj.name == "Armature" else 1,
            obj.name,
        )
    )

    return candidates[0]


def safe_normalized(vector, fallback):
    if vector.length < 1e-8:
        return fallback.copy()

    return vector.normalized()


def pose_head_world(armature, bone_name):
    pose_bone = armature.pose.bones[
        bone_name
    ]

    return (
        armature.matrix_world
        @ pose_bone.head
    )


def pose_tail_world(armature, bone_name):
    pose_bone = armature.pose.bones[
        bone_name
    ]

    return (
        armature.matrix_world
        @ pose_bone.tail
    )


def bone_direction_world(armature, bone_name):
    direction = (
        pose_tail_world(
            armature,
            bone_name,
        )
        - pose_head_world(
            armature,
            bone_name,
        )
    )

    return safe_normalized(
        direction,
        Vector((0.0, 0.0, 1.0)),
    )


def bone_rotation_world(armature, bone_name):
    pose_bone = armature.pose.bones[
        bone_name
    ]

    quaternion = (
        armature.matrix_world
        @ pose_bone.matrix
    ).to_quaternion().normalized()

    if quaternion.w < 0:
        quaternion.negate()

    return quaternion


def make_body_frame(
    left_hip,
    right_hip,
    chest,
):
    """
    Return a quaternion mapping BODY-LOCAL axes to WORLD axes.

    local +X = anatomical right
    local +Y = anatomical forward
    local +Z = anatomical up
    """

    pelvis = (
        left_hip
        + right_hip
    ) * 0.5

    right = safe_normalized(
        right_hip - left_hip,
        Vector((1.0, 0.0, 0.0)),
    )

    up = safe_normalized(
        chest - pelvis,
        Vector((0.0, 0.0, 1.0)),
    )

    forward = safe_normalized(
        right.cross(up),
        Vector((0.0, -1.0, 0.0)),
    )

    # Re-orthogonalise.
    right = safe_normalized(
        up.cross(forward),
        right,
    )

    # Matrix constructor takes rows. These values arrange the body axes
    # as columns: X=right, Y=forward, Z=up.
    matrix = Matrix(
        (
            (
                right.x,
                forward.x,
                up.x,
            ),
            (
                right.y,
                forward.y,
                up.y,
            ),
            (
                right.z,
                forward.z,
                up.z,
            ),
        )
    )

    return (
        matrix
        .to_quaternion()
        .normalized()
    )


def source_body_frame(source):
    return make_body_frame(
        left_hip=
            pose_head_world(
                source,
                "thigh_l",
            ),

        right_hip=
            pose_head_world(
                source,
                "thigh_r",
            ),

        chest=
            pose_tail_world(
                source,
                "spine_02",
            ),
    )


def target_body_frame(target):
    return make_body_frame(
        left_hip=
            pose_head_world(
                target,
                "mixamorig:LeftUpLeg",
            ),

        right_hip=
            pose_head_world(
                target,
                "mixamorig:RightUpLeg",
            ),

        chest=
            pose_tail_world(
                target,
                "mixamorig:Spine2",
            ),
    )


def world_to_armature_rotation(
    armature,
    world_quaternion,
):
    object_rotation = (
        armature
        .matrix_world
        .to_quaternion()
        .normalized()
    )

    return (
        object_rotation.inverted()
        @ world_quaternion
    ).normalized()


def reset_target_pose(target):
    for pose_bone in target.pose.bones:
        pose_bone.rotation_mode = (
            "QUATERNION"
        )

        pose_bone.location = (
            0.0,
            0.0,
            0.0,
        )

        pose_bone.rotation_quaternion = (
            Quaternion(
                (
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
        )

        pose_bone.scale = (
            1.0,
            1.0,
            1.0,
        )


def clear_previous_motionstage_actions(target):
    if target.animation_data is None:
        target.animation_data_create()

    target.animation_data.action = None

    for track in list(
        target.animation_data.nla_tracks
    ):
        target.animation_data.nla_tracks.remove(
            track
        )

    for action in list(
        bpy.data.actions
    ):
        if action.name.startswith(
            "MotionStage_Retargeted_Mixamo"
        ):
            bpy.data.actions.remove(
                action
            )


def hide_validation_collections():
    if not HIDE_MOTIONSTAGE_VALIDATION:
        return

    for name in (
        "MotionStage_19C",
        "MotionStage_19D",
    ):
        collection = bpy.data.collections.get(
            name
        )

        if collection is not None:
            collection.hide_viewport = True
            collection.hide_render = True


def validate(source, target):
    required_source = set(
        BONE_MAP.keys()
    ) | {
        "thigh_l",
        "thigh_r",
        "spine_02",
    }

    required_target = set(
        BONE_MAP.values()
    ) | {
        TARGET_REQUIRED_BONE,
        "mixamorig:LeftUpLeg",
        "mixamorig:RightUpLeg",
        "mixamorig:Spine2",
    }

    missing_source = [
        name
        for name in sorted(
            required_source
        )
        if name not in source.pose.bones
    ]

    missing_target = [
        name
        for name in sorted(
            required_target
        )
        if name not in target.pose.bones
    ]

    if missing_source:
        raise RuntimeError(
            "Missing MotionStage bones: "
            + ", ".join(
                missing_source
            )
        )

    if missing_target:
        raise RuntimeError(
            "Missing Mixamo bones: "
            + ", ".join(
                missing_target
            )
        )


def solve_rotation_only(
    target,
    bone_name,
    desired_world_rotation,
    frame,
):
    """
    Ask Blender to solve the pose orientation in the target hierarchy,
    then keep ONLY the resulting quaternion.
    """

    pose_bone = target.pose.bones[
        bone_name
    ]

    desired_armature_rotation = (
        world_to_armature_rotation(
            target,
            desired_world_rotation,
        )
    )

    current_head = (
        pose_bone.head.copy()
    )

    pose_bone.matrix = (
        Matrix.Translation(
            current_head
        )
        @ desired_armature_rotation
        .to_matrix()
        .to_4x4()
    )

    solved_rotation = (
        pose_bone
        .rotation_quaternion
        .copy()
        .normalized()
    )

    if solved_rotation.w < 0:
        solved_rotation.negate()

    # Rotation only: never bake translation or scale.
    pose_bone.location = (
        0.0,
        0.0,
        0.0,
    )

    pose_bone.scale = (
        1.0,
        1.0,
        1.0,
    )

    pose_bone.rotation_quaternion = (
        solved_rotation
    )

    pose_bone.keyframe_insert(
        data_path=
            "rotation_quaternion",
        frame=frame,
        group=bone_name,
    )

    bpy.context.view_layer.update()


# ---------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------

source = bpy.data.objects.get(
    SOURCE_ARMATURE_NAME
)

if source is None:
    raise RuntimeError(
        f"{SOURCE_ARMATURE_NAME} not found."
    )

if source.type != "ARMATURE":
    raise RuntimeError(
        f"{SOURCE_ARMATURE_NAME} is not an armature."
    )

target = find_mixamo_armature()

validate(
    source,
    target,
)

hide_validation_collections()

source.data.pose_position = "POSE"
target.data.pose_position = "POSE"

frame_start = int(
    bpy.context.scene.frame_start
)

frame_end = int(
    bpy.context.scene.frame_end
)


# ---------------------------------------------------------------------
# CAPTURE SOURCE BODY FRAME AT FRAME 1
# ---------------------------------------------------------------------

bpy.context.scene.frame_set(
    frame_start
)

bpy.context.view_layer.update()

source_body_reference = (
    source_body_frame(
        source
    )
)


# ---------------------------------------------------------------------
# CAPTURE CLEAN MIXAMO T-POSE
# ---------------------------------------------------------------------

clear_previous_motionstage_actions(
    target
)

# A previous Stage 19E.3 run may have left the armature object at a keyed
# vertical offset on the current frame. The clean Mixamo import is centred at
# the origin, so reset OBJECT translation before building the new retarget.
# Imported FBX rotation and scale remain untouched.
target.location = (
    0.0,
    0.0,
    0.0,
)

reset_target_pose(
    target
)

bpy.context.scene.frame_set(
    frame_start
)

bpy.context.view_layer.update()

target_body_reference = (
    target_body_frame(
        target
    )
)

# The diagnostic + visual validation show that the Mixamo character's visual
# facing is 180 degrees opposite to the MotionStage semantic forward direction.
# Apply the correction in TARGET BODY-LOCAL space, not Blender world space.
facing_correction_local = Quaternion(
    (0.0, 0.0, 1.0),
    radians(FACING_YAW_DEGREES),
)

# V7.2 adds a small BODY-LOCAL pitch trim after the facing correction.
# This rotates the entire retargeted character coherently rather than changing
# individual spine/leg bones.
posture_pitch_local = Quaternion(
    (1.0, 0.0, 0.0),
    radians(BODY_PITCH_DEGREES),
)

target_body_reference_corrected = (
    target_body_reference
    @ facing_correction_local
    @ posture_pitch_local
).normalized()

target_reference_direction = {
    target_name:
        bone_direction_world(
            target,
            target_name,
        )
    for target_name
    in BONE_MAP.values()
}

target_reference_world_rotation = {
    target_name:
        bone_rotation_world(
            target,
            target_name,
        )
    for target_name
    in BONE_MAP.values()
}

target_hips_reference_world_rotation = (
    bone_rotation_world(
        target,
        TARGET_REQUIRED_BONE,
    )
)


# Constant global body-coordinate calibration.
#
# Correct order:
#     target_ref = calibration @ source_ref
#
# so:
#     calibration = target_ref @ inverse(source_ref)
body_calibration = (
    target_body_reference_corrected
    @ source_body_reference.inverted()
).normalized()


action = bpy.data.actions.new(
    ACTION_NAME
)

target.animation_data.action = (
    action
)


# ---------------------------------------------------------------------
# BAKE ABSOLUTE BODY-RELATIVE MOTION
# ---------------------------------------------------------------------

for frame in range(
    frame_start,
    frame_end + 1,
):
    # Evaluate MotionStage source at this frame.
    bpy.context.scene.frame_set(
        frame
    )

    bpy.context.view_layer.update()

    source_body_current = (
        source_body_frame(
            source
        )
    )

    # Map the CURRENT source body frame into target world space.
    #
    # This is intentionally:
    #     target_current = calibration @ source_current
    #
    # NOT:
    #     source_delta @ target_reference
    target_body_current = (
        body_calibration
        @ source_body_current
    ).normalized()

    # Rotation taking target reference body frame to current body frame.
    # Compare against the ORIGINAL Mixamo T-pose body frame. This intentionally
    # includes the 180-degree facing correction in the baked target rotation.
    target_body_delta = (
        target_body_current
        @ target_body_reference.inverted()
    ).normalized()

    reset_target_pose(
        target
    )

    bpy.context.view_layer.update()

    # -------------------------------------------------------------
    # HIPS: orientation only, translation remains locked in-place.
    # -------------------------------------------------------------

    desired_hips_world_rotation = (
        target_body_delta
        @ target_hips_reference_world_rotation
    ).normalized()

    solve_rotation_only(
        target=target,
        bone_name=TARGET_REQUIRED_BONE,
        desired_world_rotation=
            desired_hips_world_rotation,
        frame=frame,
    )

    # -------------------------------------------------------------
    # TORSO + LIMBS
    #
    # CRITICAL V7 CHANGE:
    # Use the CURRENT ABSOLUTE source direction relative to the body.
    # There is NO per-bone "frame-1 neutral" correction.
    # -------------------------------------------------------------

    for target_name in TARGET_ORDER:
        source_name = TARGET_TO_SOURCE[
            target_name
        ]

        source_direction_world = (
            bone_direction_world(
                source,
                source_name,
            )
        )

        # Absolute anatomical direction relative to MotionStage body.
        source_direction_body = (
            source_body_current.inverted()
            @ source_direction_world
        ).normalized()

        # Interpret those same anatomical body coordinates on Mixamo.
        desired_target_direction_world = (
            target_body_current
            @ source_direction_body
        ).normalized()

        # Carry Mixamo's neutral bone roll with the body's global motion.
        baseline_target_direction_world = (
            target_body_delta
            @ target_reference_direction[
                target_name
            ]
        ).normalized()

        baseline_target_rotation_world = (
            target_body_delta
            @ target_reference_world_rotation[
                target_name
            ]
        ).normalized()

        # Swing only: point the Mixamo bone at the desired anatomical
        # direction while preserving its neutral axial roll as much as possible.
        swing_world = (
            baseline_target_direction_world
            .rotation_difference(
                desired_target_direction_world
            )
            .normalized()
        )

        desired_target_rotation_world = (
            swing_world
            @ baseline_target_rotation_world
        ).normalized()

        solve_rotation_only(
            target=target,
            bone_name=target_name,
            desired_world_rotation=
                desired_target_rotation_world,
            frame=frame,
        )


# ---------------------------------------------------------------------
# FINISH
# ---------------------------------------------------------------------

bpy.context.scene.frame_set(
    frame_start
)

bpy.context.view_layer.update()

bpy.ops.object.select_all(
    action="DESELECT"
)

target.select_set(
    True
)

bpy.context.view_layer.objects.active = (
    target
)

target.show_in_front = True

print()
print("=" * 82)

print(
    "MotionStage Stage 19E.2 - "
    "Mixamo Absolute Body-Space Retarget V7.3 Foot-Aware"
)

print("=" * 82)

print(
    "Source rig       :",
    source.name,
)

print(
    "Target rig       :",
    target.name,
)

print(
    "Frames baked     :",
    frame_end - frame_start + 1,
)

print(
    "Driven segments  :",
    len(BONE_MAP),
)

print(
    "Facing correction:",
    f"{FACING_YAW_DEGREES:.0f} degrees around target local up"
)

print(
    "Posture pitch    :",
    f"{BODY_PITCH_DEGREES:.1f} degrees around target local right axis"
)

print(
    "Root translation : none"
)

print(
    "Scale animation  : none"
)

print(
    "Limb calibration : ABSOLUTE body-relative directions"
)

print(
    "Frame 1 role     : body-coordinate alignment only"
)

print(
    "Per-bone frame-1 neutral correction : REMOVED"
)

print()
print(
    "This version keeps the successful V7.2 body calibration and directly "
    "retargets MotionStage foot_l / foot_r onto the Mixamo feet."
)

print(
    "Mixamo toes remain inherited from the foot bones because MotionStage "
    "does not currently expose a separate toe segment."
)

print("=" * 82)
