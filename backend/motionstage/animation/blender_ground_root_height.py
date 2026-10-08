# MotionStage Stage 19E.4 — SUPPORT-AWARE GROUND / ROOT-HEIGHT CORRECTION V2
#
# Run INSIDE Blender AFTER:
#   blender_retarget_mixamo_v7_3_foot_aware.py
#
# Why V2:
# The previous grounding pass always grounded whichever TARGET foot happened
# to be lowest. That can disagree with the MotionStage reference and subtly
# distort the lower-body read.
#
# V2 lets the SOURCE MotionStage rig decide which foot is the support foot:
#   - if one source foot is clearly lower, ground that corresponding Mixamo foot
#   - if both source feet are at similar height, use double-support grounding
#
# The non-support foot is free to lift naturally.
#
# This pass changes ONLY Mixamo armature OBJECT Z translation.
# It does NOT change V7.3 bone rotations, X/Y translation, or scale.

import bpy


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

SOURCE_ARMATURE_NAME = "MotionStage_RetargetRig"
TARGET_REQUIRED_BONE = "mixamorig:Hips"

EXPECTED_ACTION_PREFIX = "MotionStage_Retargeted_Mixamo_V7_3"

FLOOR_OBJECT_NAME = "MotionStage_Floor"

# Used only if the floor object cannot be found.
FALLBACK_GROUND_Z = -1.0

# Small sole clearance.
GROUND_CLEARANCE = 0.003

# Reference-foot height difference needed to call one side the support foot.
# MotionStage coordinates are approximately metre-scale.
SUPPORT_HEIGHT_THRESHOLD = 0.035

# Ignore very weak foot/toe skin weights.
FOOT_WEIGHT_THRESHOLD = 0.05

LEFT_FOOT_GROUPS = {
    "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase",
}

RIGHT_FOOT_GROUPS = {
    "mixamorig:RightFoot",
    "mixamorig:RightToeBase",
}


# ---------------------------------------------------------------------
# DISCOVERY
# ---------------------------------------------------------------------

def find_mixamo_armature():
    candidates = []

    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue

        if TARGET_REQUIRED_BONE in obj.data.bones:
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(
            "No Mixamo armature found. Expected "
            f"'{TARGET_REQUIRED_BONE}'."
        )

    candidates.sort(
        key=lambda obj: (
            0 if obj.name == "Armature" else 1,
            obj.name,
        )
    )

    return candidates[0]


def uses_armature(mesh_obj, armature_obj):
    if mesh_obj.type != "MESH":
        return False

    for modifier in mesh_obj.modifiers:
        if (
            modifier.type == "ARMATURE"
            and modifier.object == armature_obj
        ):
            return True

    parent = mesh_obj.parent

    while parent is not None:
        if parent == armature_obj:
            return True
        parent = parent.parent

    return False


def character_meshes(armature):
    meshes = [
        obj
        for obj in bpy.data.objects
        if uses_armature(
            obj,
            armature,
        )
    ]

    if not meshes:
        raise RuntimeError(
            "Mixamo armature found, but no skinned character meshes "
            "could be associated with it."
        )

    return meshes


# ---------------------------------------------------------------------
# FOOT VERTEX CACHE
# ---------------------------------------------------------------------

def weighted_vertex_indices(
    mesh_obj,
    group_names,
):
    wanted = {
        group.index
        for group in mesh_obj.vertex_groups
        if group.name in group_names
    }

    if not wanted:
        return []

    indices = []

    for vertex in mesh_obj.data.vertices:
        for membership in vertex.groups:
            if (
                membership.group in wanted
                and membership.weight >= FOOT_WEIGHT_THRESHOLD
            ):
                indices.append(
                    vertex.index
                )
                break

    return indices


def build_vertex_cache(meshes):
    return {
        "left": {
            mesh.name:
                weighted_vertex_indices(
                    mesh,
                    LEFT_FOOT_GROUPS,
                )
            for mesh in meshes
        },

        "right": {
            mesh.name:
                weighted_vertex_indices(
                    mesh,
                    RIGHT_FOOT_GROUPS,
                )
            for mesh in meshes
        },
    }


# ---------------------------------------------------------------------
# HEIGHT HELPERS
# ---------------------------------------------------------------------

def lowest_weighted_side_z(
    meshes,
    side_cache,
    depsgraph,
):
    values = []

    for mesh_obj in meshes:
        indices = side_cache.get(
            mesh_obj.name,
            [],
        )

        if not indices:
            continue

        evaluated_obj = mesh_obj.evaluated_get(
            depsgraph
        )

        evaluated_mesh = evaluated_obj.to_mesh()

        try:
            matrix_world = (
                evaluated_obj.matrix_world
            )

            count = len(
                evaluated_mesh.vertices
            )

            for index in indices:
                if index >= count:
                    continue

                world_position = (
                    matrix_world
                    @ evaluated_mesh
                    .vertices[index]
                    .co
                )

                values.append(
                    float(
                        world_position.z
                    )
                )

        finally:
            evaluated_obj.to_mesh_clear()

    if not values:
        return None

    return min(
        values
    )


def target_foot_bone_z(
    armature,
    side,
):
    name = (
        "mixamorig:LeftFoot"
        if side == "left"
        else "mixamorig:RightFoot"
    )

    pose_bone = armature.pose.bones.get(
        name
    )

    if pose_bone is None:
        raise RuntimeError(
            f"Missing target foot bone: {name}"
        )

    head_z = (
        armature.matrix_world
        @ pose_bone.head
    ).z

    tail_z = (
        armature.matrix_world
        @ pose_bone.tail
    ).z

    return float(
        min(
            head_z,
            tail_z,
        )
    )


def source_foot_z(
    source,
    side,
):
    name = (
        "foot_l"
        if side == "left"
        else "foot_r"
    )

    pose_bone = source.pose.bones.get(
        name
    )

    if pose_bone is None:
        raise RuntimeError(
            f"Missing MotionStage source foot bone: {name}"
        )

    head_z = (
        source.matrix_world
        @ pose_bone.head
    ).z

    tail_z = (
        source.matrix_world
        @ pose_bone.tail
    ).z

    return float(
        min(
            head_z,
            tail_z,
        )
    )


def get_ground_z():
    floor = bpy.data.objects.get(
        FLOOR_OBJECT_NAME
    )

    if floor is not None:
        return (
            float(
                floor.matrix_world
                .translation.z
            ),
            f"object:{FLOOR_OBJECT_NAME}",
        )

    return (
        float(
            FALLBACK_GROUND_Z
        ),
        "fallback",
    )


# ---------------------------------------------------------------------
# EXISTING ROOT-HEIGHT CLEANUP
# ---------------------------------------------------------------------

def remove_object_location_curves(
    armature,
):
    animation_data = armature.animation_data

    if (
        animation_data is None
        or animation_data.action is None
        or not hasattr(
            animation_data.action,
            "fcurves",
        )
    ):
        return 0

    action = animation_data.action

    curves = [
        curve
        for curve in action.fcurves
        if curve.data_path == "location"
    ]

    for curve in curves:
        action.fcurves.remove(
            curve
        )

    return len(
        curves
    )


def make_root_height_linear(
    armature,
):
    animation_data = armature.animation_data

    if (
        animation_data is None
        or animation_data.action is None
        or not hasattr(
            animation_data.action,
            "fcurves",
        )
    ):
        return

    for curve in animation_data.action.fcurves:
        if (
            curve.data_path == "location"
            and curve.array_index == 2
        ):
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

source = bpy.data.objects.get(
    SOURCE_ARMATURE_NAME
)

if source is None:
    raise RuntimeError(
        f"{SOURCE_ARMATURE_NAME} not found."
    )

target = find_mixamo_armature()

meshes = character_meshes(
    target
)

if (
    target.animation_data is None
    or target.animation_data.action is None
):
    raise RuntimeError(
        "Mixamo target has no active retarget action. "
        "Run Stage 19E.4 V7.3 first."
    )

action = target.animation_data.action

if not action.name.startswith(
    EXPECTED_ACTION_PREFIX
):
    print(
        "WARNING: active action is",
        action.name,
    )
    print(
        "Expected prefix:",
        EXPECTED_ACTION_PREFIX,
    )


ground_z, ground_source = (
    get_ground_z()
)

vertex_cache = build_vertex_cache(
    meshes
)

left_cached = sum(
    len(indices)
    for indices
    in vertex_cache["left"].values()
)

right_cached = sum(
    len(indices)
    for indices
    in vertex_cache["right"].values()
)

removed_curves = (
    remove_object_location_curves(
        target
    )
)

frame_start = int(
    bpy.context.scene.frame_start
)

frame_end = int(
    bpy.context.scene.frame_end
)

# Reset to the clean imported object origin before measurement.
target.location = (
    0.0,
    0.0,
    0.0,
)

base_location = (
    target.location.copy()
)

depsgraph = (
    bpy.context.evaluated_depsgraph_get()
)

records = []


# ---------------------------------------------------------------------
# MEASURE SUPPORT FOOT + REQUIRED TARGET ROOT HEIGHT
# ---------------------------------------------------------------------

for frame in range(
    frame_start,
    frame_end + 1,
):
    bpy.context.scene.frame_set(
        frame
    )

    target.location = (
        base_location.copy()
    )

    bpy.context.view_layer.update()

    source_left_z = (
        source_foot_z(
            source,
            "left",
        )
    )

    source_right_z = (
        source_foot_z(
            source,
            "right",
        )
    )

    source_delta = (
        source_left_z
        - source_right_z
    )

    if (
        source_delta
        < -SUPPORT_HEIGHT_THRESHOLD
    ):
        support = "left"

    elif (
        source_delta
        > SUPPORT_HEIGHT_THRESHOLD
    ):
        support = "right"

    else:
        support = "both"


    target_left_z = (
        lowest_weighted_side_z(
            meshes,
            vertex_cache["left"],
            depsgraph,
        )
    )

    target_right_z = (
        lowest_weighted_side_z(
            meshes,
            vertex_cache["right"],
            depsgraph,
        )
    )

    if target_left_z is None:
        target_left_z = (
            target_foot_bone_z(
                target,
                "left",
            )
        )

    if target_right_z is None:
        target_right_z = (
            target_foot_bone_z(
                target,
                "right",
            )
        )


    if support == "left":
        support_target_z = (
            target_left_z
        )

    elif support == "right":
        support_target_z = (
            target_right_z
        )

    else:
        # Double support: keep whichever sole would otherwise penetrate first.
        support_target_z = min(
            target_left_z,
            target_right_z,
        )


    desired_support_z = (
        ground_z
        + GROUND_CLEARANCE
    )

    z_offset = (
        desired_support_z
        - support_target_z
    )

    records.append(
        {
            "frame": frame,
            "support": support,
            "source_left_z": source_left_z,
            "source_right_z": source_right_z,
            "target_left_z": target_left_z,
            "target_right_z": target_right_z,
            "z_offset": float(
                z_offset
            ),
        }
    )


# ---------------------------------------------------------------------
# BAKE ROOT Z ONLY
# ---------------------------------------------------------------------

for record in records:
    frame = record[
        "frame"
    ]

    bpy.context.scene.frame_set(
        frame
    )

    target.location.x = (
        base_location.x
    )

    target.location.y = (
        base_location.y
    )

    target.location.z = (
        base_location.z
        + record[
            "z_offset"
        ]
    )

    target.keyframe_insert(
        data_path="location",
        index=2,
        frame=frame,
        group=
            "MotionStage Support-Aware Root Height",
    )


make_root_height_linear(
    target
)


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

support_counts = {
    "left": 0,
    "right": 0,
    "both": 0,
}

for record in records:
    support_counts[
        record["support"]
    ] += 1

offsets = [
    record[
        "z_offset"
    ]
    for record in records
]


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


print()
print("=" * 84)

print(
    "MotionStage Stage 19E.4 - "
    "Support-Aware Ground / Root-Height V2"
)

print("=" * 84)

print(
    "Source rig           :",
    source.name,
)

print(
    "Target rig           :",
    target.name,
)

print(
    "Active action        :",
    action.name,
)

print(
    "Ground source        :",
    ground_source,
)

print(
    "Ground Z             :",
    f"{ground_z:.4f}",
)

print(
    "Left foot vertices   :",
    left_cached,
)

print(
    "Right foot vertices  :",
    right_cached,
)

print(
    "Left-support frames  :",
    support_counts["left"],
)

print(
    "Right-support frames :",
    support_counts["right"],
)

print(
    "Double-support frames:",
    support_counts["both"],
)

print(
    "Min root-Z correction:",
    f"{min(offsets):.4f}",
)

print(
    "Max root-Z correction:",
    f"{max(offsets):.4f}",
)

print(
    "Old location curves  :",
    removed_curves,
    "removed",
)

print()

print(
    "Only the Mixamo armature OBJECT Z translation was keyed."
)

print(
    "The non-support foot is free to lift according to the V7.3 foot pose."
)

print("=" * 84)
