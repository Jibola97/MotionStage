# MotionStage Stage 19D
# Build a single skinned humanoid mesh driven by MotionStage_RetargetRig.
#
# Run INSIDE Blender after Stage 19B / 19C:
#   1. Keep the current scene open
#   2. Go to Scripting
#   3. Open this file
#   4. Press Run Script
#
# Creates:
#   MotionStage_19D
#   MotionStage_SkinnedCharacter
#
# Unlike the Stage 19C proxy, this stage joins the visible body into ONE mesh
# object and drives it with an Armature modifier + vertex groups.
#
# The weights are deliberately deterministic and rigid per body section.
# This validates the standard Blender skinned-mesh workflow before moving to
# a production/external humanoid character in the next stage.

import bpy
from mathutils import Vector


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

ARMATURE_NAME = "MotionStage_RetargetRig"
COLLECTION_NAME = "MotionStage_19D"
CHARACTER_NAME = "MotionStage_SkinnedCharacter"

HIDE_STAGE19C_PROXY = True

# Body proportions in metres, tuned for the current MotionStage bridge rig.
RADIUS = {
    "pelvis": 0.115,
    "spine_01": 0.105,
    "spine_02": 0.125,
    "neck_01": 0.040,

    "clavicle_l": 0.036,
    "upperarm_l": 0.055,
    "lowerarm_l": 0.045,

    "clavicle_r": 0.036,
    "upperarm_r": 0.055,
    "lowerarm_r": 0.045,

    "thigh_l": 0.072,
    "calf_l": 0.058,

    "thigh_r": 0.072,
    "calf_r": 0.058,
}

HEAD_RADIUS = 0.115
HAND_RADIUS = 0.055

FOOT_WIDTH = 0.11
FOOT_HEIGHT = 0.075

SEGMENT_LENGTH_SCALE = 0.98


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


def rest_bone(armature, name):
    bone = armature.data.bones.get(name)

    if bone is None:
        raise KeyError(
            f"Required bone '{name}' was not found on {ARMATURE_NAME}."
        )

    return bone


def safe_direction(head, tail):
    direction = tail - head

    if direction.length < 1e-7:
        return Vector((0.0, 0.05, 0.0))

    return direction


def orient_between(obj, head, tail):
    direction = safe_direction(
        head,
        tail,
    )

    obj.location = (
        head + tail
    ) * 0.5

    # Blender cylinder primitive uses local Z.
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (
        direction
        .normalized()
        .to_track_quat(
            "Z",
            "Y",
        )
    )


def assign_all_vertices_to_group(
    obj,
    bone_name,
):
    group = obj.vertex_groups.new(
        name=bone_name
    )

    group.add(
        list(range(len(obj.data.vertices))),
        1.0,
        "REPLACE",
    )


def add_cylinder_part(
    collection,
    armature,
    bone_name,
    display_name,
    radius,
):
    bone = rest_bone(
        armature,
        bone_name,
    )

    head = bone.head_local.copy()
    tail = bone.tail_local.copy()

    direction = safe_direction(
        head,
        tail,
    )

    length = max(
        direction.length
        * SEGMENT_LENGTH_SCALE,
        radius * 2.0,
    )

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=radius,
        depth=length,
        location=(0.0, 0.0, 0.0),
    )

    obj = bpy.context.object
    obj.name = display_name

    orient_between(
        obj,
        head,
        tail,
    )

    bpy.ops.object.transform_apply(
        location=True,
        rotation=True,
        scale=True,
    )

    assign_all_vertices_to_group(
        obj,
        bone_name,
    )

    relink_object(
        obj,
        collection,
    )

    return obj


def add_sphere_part(
    collection,
    armature,
    bone_name,
    display_name,
    radius,
    at_tail=True,
):
    bone = rest_bone(
        armature,
        bone_name,
    )

    location = (
        bone.tail_local.copy()
        if at_tail
        else bone.head_local.copy()
    )

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=28,
        ring_count=16,
        radius=radius,
        location=location,
    )

    obj = bpy.context.object
    obj.name = display_name

    bpy.ops.object.transform_apply(
        location=True,
        rotation=True,
        scale=True,
    )

    assign_all_vertices_to_group(
        obj,
        bone_name,
    )

    relink_object(
        obj,
        collection,
    )

    return obj


def add_foot_part(
    collection,
    armature,
    bone_name,
    display_name,
):
    bone = rest_bone(
        armature,
        bone_name,
    )

    head = bone.head_local.copy()
    tail = bone.tail_local.copy()

    direction = safe_direction(
        head,
        tail,
    )

    length = max(
        direction.length
        * SEGMENT_LENGTH_SCALE,
        0.12,
    )

    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, 0.0, 0.0),
    )

    obj = bpy.context.object
    obj.name = display_name

    # Local Z will be aligned to the foot-bone direction.
    obj.dimensions = (
        FOOT_WIDTH,
        FOOT_HEIGHT,
        length,
    )

    orient_between(
        obj,
        head,
        tail,
    )

    bpy.ops.object.transform_apply(
        location=True,
        rotation=True,
        scale=True,
    )

    assign_all_vertices_to_group(
        obj,
        bone_name,
    )

    relink_object(
        obj,
        collection,
    )

    return obj


def join_parts(parts):
    if not parts:
        raise RuntimeError(
            "No character body parts were created."
        )

    bpy.ops.object.select_all(
        action="DESELECT"
    )

    for obj in parts:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = (
        parts[0]
    )

    bpy.ops.object.join()

    joined = bpy.context.object
    joined.name = CHARACTER_NAME

    return joined


def create_material():
    name = "MotionStage_19D_Material"

    material = bpy.data.materials.get(
        name
    )

    if material is None:
        material = bpy.data.materials.new(
            name
        )

    material.diffuse_color = (
        0.12,
        0.55,
        0.82,
        1.0,
    )

    material.metallic = 0.03
    material.roughness = 0.42

    return material


def add_armature_modifier(
    mesh_object,
    armature,
):
    modifier = mesh_object.modifiers.new(
        name="MotionStage Armature",
        type="ARMATURE",
    )

    modifier.object = armature
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False

    return modifier


def hide_old_proxy():
    if not HIDE_STAGE19C_PROXY:
        return

    collection = bpy.data.collections.get(
        "MotionStage_19C"
    )

    if collection is not None:
        collection.hide_viewport = True
        collection.hide_render = True


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

armature = bpy.data.objects.get(
    ARMATURE_NAME
)

if armature is None:
    raise RuntimeError(
        f"{ARMATURE_NAME} was not found. Run Stage 19B first."
    )

if armature.type != "ARMATURE":
    raise RuntimeError(
        f"{ARMATURE_NAME} exists but is not an armature."
    )

# Build against the rig's rest pose / first animation frame.
bpy.context.scene.frame_set(1)

remove_existing_collection(
    COLLECTION_NAME
)

collection = create_collection(
    COLLECTION_NAME
)

parts = []


# ---------------------------------------------------------------------
# TORSO
# ---------------------------------------------------------------------

for bone_name in (
    "pelvis",
    "spine_01",
    "spine_02",
    "neck_01",
):
    parts.append(
        add_cylinder_part(
            collection,
            armature,
            bone_name,
            f"{CHARACTER_NAME}_{bone_name}",
            RADIUS[bone_name],
        )
    )

parts.append(
    add_sphere_part(
        collection,
        armature,
        "head",
        f"{CHARACTER_NAME}_head",
        HEAD_RADIUS,
        at_tail=True,
    )
)


# ---------------------------------------------------------------------
# ARMS
# ---------------------------------------------------------------------

for bone_name in (
    "clavicle_l",
    "upperarm_l",
    "lowerarm_l",
    "clavicle_r",
    "upperarm_r",
    "lowerarm_r",
):
    parts.append(
        add_cylinder_part(
            collection,
            armature,
            bone_name,
            f"{CHARACTER_NAME}_{bone_name}",
            RADIUS[bone_name],
        )
    )

parts.append(
    add_sphere_part(
        collection,
        armature,
        "hand_l",
        f"{CHARACTER_NAME}_hand_l",
        HAND_RADIUS,
        at_tail=True,
    )
)

parts.append(
    add_sphere_part(
        collection,
        armature,
        "hand_r",
        f"{CHARACTER_NAME}_hand_r",
        HAND_RADIUS,
        at_tail=True,
    )
)


# ---------------------------------------------------------------------
# LEGS
# ---------------------------------------------------------------------

for bone_name in (
    "thigh_l",
    "calf_l",
    "thigh_r",
    "calf_r",
):
    parts.append(
        add_cylinder_part(
            collection,
            armature,
            bone_name,
            f"{CHARACTER_NAME}_{bone_name}",
            RADIUS[bone_name],
        )
    )

parts.append(
    add_foot_part(
        collection,
        armature,
        "foot_l",
        f"{CHARACTER_NAME}_foot_l",
    )
)

parts.append(
    add_foot_part(
        collection,
        armature,
        "foot_r",
        f"{CHARACTER_NAME}_foot_r",
    )
)


# ---------------------------------------------------------------------
# JOIN INTO ONE SKINNED MESH
# ---------------------------------------------------------------------

character = join_parts(
    parts
)

# Put mesh object into the same transform space as the retarget armature.
character.matrix_world = (
    armature.matrix_world.copy()
)

material = create_material()

character.data.materials.clear()
character.data.materials.append(
    material
)

for polygon in character.data.polygons:
    polygon.use_smooth = True

add_armature_modifier(
    character,
    armature,
)

# Optional light smoothing of the visible mesh without changing weights.
bevel = character.modifiers.new(
    name="MotionStage Surface Bevel",
    type="BEVEL",
)

bevel.width = 0.008
bevel.segments = 2

hide_old_proxy()

armature.show_in_front = True

bpy.context.view_layer.update()

bpy.ops.object.select_all(
    action="DESELECT"
)

character.select_set(True)

bpy.context.view_layer.objects.active = (
    character
)

print()
print("=" * 72)
print(
    "MotionStage Stage 19D - "
    "Single Skinned Humanoid Mesh"
)
print("=" * 72)
print(
    "Driver rig       :",
    ARMATURE_NAME,
)
print(
    "Character        :",
    CHARACTER_NAME,
)
print(
    "Mesh objects     : 1",
)
print(
    "Vertex groups    :",
    len(
        character.vertex_groups
    ),
)
print(
    "Armature modifier:",
    "yes",
)
print(
    "Frames           :",
    bpy.context.scene.frame_end,
)
print()
print(
    "Press Space to play. "
    "The single blue mesh should follow "
    "MotionStage_RetargetRig."
)
print(
    "This is the first standard skinned-mesh "
    "validation stage; production character "
    "retargeting comes next."
)
print("=" * 72)
