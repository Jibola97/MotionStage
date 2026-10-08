# MotionStage Stage 19C — FIXED
# Visible humanoid proxy driven by MotionStage_RetargetRig.
#
# This version uses bone constraints instead of bone-parenting offsets.
# It fixes the scattered/exploded body-part issue seen in the first 19C test.
#
# Run INSIDE Blender after Stage 19B:
#   1. Keep the current Stage 19B scene open
#   2. Go to Scripting
#   3. Open this file
#   4. Press Run Script
#
# It removes/recreates the MotionStage_19C collection.

import bpy
from math import radians
from mathutils import Matrix


ARMATURE_NAME = "MotionStage_RetargetRig"
COLLECTION_NAME = "MotionStage_19C"
CHARACTER_NAME = "MotionStage_ProxyCharacter"


# ---------------------------------------------------------------------
# VISUAL PROPORTIONS
# ---------------------------------------------------------------------

TORSO_RADIUS = 0.12
PELVIS_RADIUS = 0.11

UPPER_ARM_RADIUS = 0.050
FOREARM_RADIUS = 0.042

THIGH_RADIUS = 0.065
CALF_RADIUS = 0.052

HAND_RADIUS = 0.050
HEAD_RADIUS = 0.105

FOOT_WIDTH = 0.105
FOOT_HEIGHT = 0.070

SEGMENT_LENGTH_SCALE = 0.92


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def remove_existing_collection(name):
    collection = bpy.data.collections.get(name)

    if collection is None:
        return

    for obj in list(collection.objects):
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


def relink_object(obj, collection):
    for source_collection in list(
        obj.users_collection
    ):
        source_collection.objects.unlink(
            obj
        )

    collection.objects.link(
        obj
    )


def get_rest_bone(armature, bone_name):
    bone = armature.data.bones.get(
        bone_name
    )

    if bone is None:
        raise KeyError(
            f"Bone '{bone_name}' was not found "
            f"on {ARMATURE_NAME}."
        )

    return bone


def add_copy_transforms_constraint(
    obj,
    armature,
    bone_name,
):
    constraint = obj.constraints.new(
        type="COPY_TRANSFORMS"
    )

    constraint.name = (
        "MotionStage Bone Driver"
    )

    constraint.target = armature
    constraint.subtarget = bone_name

    # With a bone subtarget, WORLD gives us the evaluated bone transform
    # in world space. The proxy object itself stays at identity.
    constraint.target_space = "WORLD"
    constraint.owner_space = "WORLD"

    return constraint


def add_bone_tail_location_constraint(
    obj,
    armature,
    bone_name,
):
    location_constraint = obj.constraints.new(
        type="COPY_LOCATION"
    )

    location_constraint.name = (
        "MotionStage Bone Tail"
    )

    location_constraint.target = armature
    location_constraint.subtarget = (
        bone_name
    )

    # 1.0 means bone tail rather than bone head.
    location_constraint.head_tail = 1.0

    rotation_constraint = obj.constraints.new(
        type="COPY_ROTATION"
    )

    rotation_constraint.name = (
        "MotionStage Bone Rotation"
    )

    rotation_constraint.target = armature
    rotation_constraint.subtarget = (
        bone_name
    )

    return (
        location_constraint,
        rotation_constraint,
    )


def material_for_character():
    name = (
        "MotionStage_ProxyCharacter_Material"
    )

    material = bpy.data.materials.get(
        name
    )

    if material is None:
        material = bpy.data.materials.new(
            name
        )

        material.diffuse_color = (
            0.07,
            0.38,
            0.78,
            1.0,
        )

        material.metallic = 0.04
        material.roughness = 0.45

    return material


def assign_material(obj, material):
    if obj.type != "MESH":
        return

    if len(obj.data.materials) == 0:
        obj.data.materials.append(
            material
        )
    else:
        obj.data.materials[0] = (
            material
        )


def smooth_object(obj):
    if obj.type != "MESH":
        return

    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def create_bone_cylinder(
    collection,
    armature,
    bone_name,
    object_name,
    radius,
    material,
):
    bone = get_rest_bone(
        armature,
        bone_name,
    )

    rest_length = max(
        float(bone.length),
        0.02,
    )

    visual_length = (
        rest_length
        * SEGMENT_LENGTH_SCALE
    )

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20,
        radius=radius,
        depth=visual_length,
        location=(0.0, 0.0, 0.0),
    )

    obj = bpy.context.object
    obj.name = object_name

    # Primitive cylinders are along local Z.
    # MotionStage/Blender bones point along local +Y.
    obj.data.transform(
        Matrix.Rotation(
            radians(-90.0),
            4,
            "X",
        )
    )

    # Put the mesh centre halfway along +Y while keeping the
    # OBJECT ORIGIN at the bone head.
    obj.data.transform(
        Matrix.Translation(
            (
                0.0,
                visual_length / 2.0,
                0.0,
            )
        )
    )

    relink_object(
        obj,
        collection,
    )

    assign_material(
        obj,
        material,
    )

    smooth_object(
        obj
    )

    add_copy_transforms_constraint(
        obj,
        armature,
        bone_name,
    )

    return obj


def create_hand_sphere(
    collection,
    armature,
    bone_name,
    object_name,
    radius,
    material,
):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        radius=radius,
        location=(0.0, 0.0, 0.0),
    )

    obj = bpy.context.object
    obj.name = object_name

    relink_object(
        obj,
        collection,
    )

    assign_material(
        obj,
        material,
    )

    smooth_object(
        obj
    )

    # Follow the TAIL of the hand bone.
    add_bone_tail_location_constraint(
        obj,
        armature,
        bone_name,
    )

    return obj


def create_head_sphere(
    collection,
    armature,
    material,
):
    return create_hand_sphere(
        collection=collection,
        armature=armature,
        bone_name="head",
        object_name=(
            f"{CHARACTER_NAME}_Head"
        ),
        radius=HEAD_RADIUS,
        material=material,
    )


def create_foot_box(
    collection,
    armature,
    bone_name,
    object_name,
    material,
):
    bone = get_rest_bone(
        armature,
        bone_name,
    )

    rest_length = max(
        float(bone.length),
        0.04,
    )

    visual_length = (
        rest_length
        * SEGMENT_LENGTH_SCALE
    )

    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, 0.0, 0.0),
    )

    obj = bpy.context.object
    obj.name = object_name

    # Model the foot directly in BONE LOCAL SPACE:
    # X = width, Y = along the foot bone, Z = height.
    obj.data.transform(
        Matrix.Diagonal(
            (
                FOOT_WIDTH,
                visual_length,
                FOOT_HEIGHT,
                1.0,
            )
        )
    )

    obj.data.transform(
        Matrix.Translation(
            (
                0.0,
                visual_length / 2.0,
                0.0,
            )
        )
    )

    relink_object(
        obj,
        collection,
    )

    assign_material(
        obj,
        material,
    )

    add_copy_transforms_constraint(
        obj,
        armature,
        bone_name,
    )

    return obj


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

armature = bpy.data.objects.get(
    ARMATURE_NAME
)

if armature is None:
    raise RuntimeError(
        f"{ARMATURE_NAME} not found. "
        "Run Stage 19B first."
    )

if armature.type != "ARMATURE":
    raise RuntimeError(
        f"{ARMATURE_NAME} exists but is "
        "not an armature."
    )

# Always build the proxy from frame 1 so the first visual validation
# starts from a deterministic pose.
bpy.context.scene.frame_set(1)

remove_existing_collection(
    COLLECTION_NAME
)

collection = create_collection(
    COLLECTION_NAME
)

material = material_for_character()

parts = []


# ---------------------------------------------------------------------
# TORSO
# ---------------------------------------------------------------------

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "pelvis",
        f"{CHARACTER_NAME}_Pelvis",
        PELVIS_RADIUS,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "spine_01",
        f"{CHARACTER_NAME}_Spine01",
        TORSO_RADIUS * 0.82,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "spine_02",
        f"{CHARACTER_NAME}_Chest",
        TORSO_RADIUS,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "neck_01",
        f"{CHARACTER_NAME}_Neck",
        UPPER_ARM_RADIUS * 0.58,
        material,
    )
)

parts.append(
    create_head_sphere(
        collection,
        armature,
        material,
    )
)


# ---------------------------------------------------------------------
# LEFT ARM
# ---------------------------------------------------------------------

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "clavicle_l",
        f"{CHARACTER_NAME}_Clavicle_L",
        UPPER_ARM_RADIUS * 0.52,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "upperarm_l",
        f"{CHARACTER_NAME}_UpperArm_L",
        UPPER_ARM_RADIUS,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "lowerarm_l",
        f"{CHARACTER_NAME}_Forearm_L",
        FOREARM_RADIUS,
        material,
    )
)

parts.append(
    create_hand_sphere(
        collection,
        armature,
        "hand_l",
        f"{CHARACTER_NAME}_Hand_L",
        HAND_RADIUS,
        material,
    )
)


# ---------------------------------------------------------------------
# RIGHT ARM
# ---------------------------------------------------------------------

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "clavicle_r",
        f"{CHARACTER_NAME}_Clavicle_R",
        UPPER_ARM_RADIUS * 0.52,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "upperarm_r",
        f"{CHARACTER_NAME}_UpperArm_R",
        UPPER_ARM_RADIUS,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "lowerarm_r",
        f"{CHARACTER_NAME}_Forearm_R",
        FOREARM_RADIUS,
        material,
    )
)

parts.append(
    create_hand_sphere(
        collection,
        armature,
        "hand_r",
        f"{CHARACTER_NAME}_Hand_R",
        HAND_RADIUS,
        material,
    )
)


# ---------------------------------------------------------------------
# LEFT LEG
# ---------------------------------------------------------------------

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "thigh_l",
        f"{CHARACTER_NAME}_Thigh_L",
        THIGH_RADIUS,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "calf_l",
        f"{CHARACTER_NAME}_Calf_L",
        CALF_RADIUS,
        material,
    )
)

parts.append(
    create_foot_box(
        collection,
        armature,
        "foot_l",
        f"{CHARACTER_NAME}_Foot_L",
        material,
    )
)


# ---------------------------------------------------------------------
# RIGHT LEG
# ---------------------------------------------------------------------

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "thigh_r",
        f"{CHARACTER_NAME}_Thigh_R",
        THIGH_RADIUS,
        material,
    )
)

parts.append(
    create_bone_cylinder(
        collection,
        armature,
        "calf_r",
        f"{CHARACTER_NAME}_Calf_R",
        CALF_RADIUS,
        material,
    )
)

parts.append(
    create_foot_box(
        collection,
        armature,
        "foot_r",
        f"{CHARACTER_NAME}_Foot_R",
        material,
    )
)


# ---------------------------------------------------------------------
# FINISH
# ---------------------------------------------------------------------

# Keep the rig visible for this validation pass.
armature.show_in_front = True

# Force constraint evaluation.
bpy.context.view_layer.update()

# Select the new character pieces so they are easy to frame.
bpy.ops.object.select_all(
    action="DESELECT"
)

for obj in parts:
    obj.select_set(True)

if parts:
    bpy.context.view_layer.objects.active = (
        parts[0]
    )

print()
print("=" * 72)
print(
    "MotionStage Stage 19C - "
    "Proxy Humanoid Character (FIXED)"
)
print("=" * 72)
print(
    "Driver rig        :",
    ARMATURE_NAME,
)
print(
    "Character         :",
    CHARACTER_NAME,
)
print(
    "Character pieces  :",
    len(parts),
)
print(
    "Current frame     :",
    bpy.context.scene.frame_current,
)
print()
print(
    "The proxy now uses bone constraints "
    "rather than offset bone-parenting."
)
print(
    "Press Space to play. The blue body "
    "segments should stay attached to "
    "MotionStage_RetargetRig."
)
print("=" * 72)
