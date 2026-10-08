# MotionStage Stage 19F
# FINAL UNREAL FBX EXPORT
#
# Purpose
# -------
# Export the validated Stage 19E result to a clean FBX for Unreal Engine.
#
# IMPORTANT:
# The Stage 19E Blender scene contains many diagnostic / calibration duplicate
# meshes. The old exporter collected every mesh skinned to the Mixamo armature,
# which is no longer safe. This version exports ONLY:
#
#   1. the Mixamo armature
#   2. Alpha_Surface_MS19E7D_KneeCalibrated
#   3. Alpha_Joints (when present and skinned to the same armature)
#
# The final knee correction is driven by animated shape keys, so this exporter
# deliberately avoids applying mesh modifiers during FBX export. Applying mesh
# modifiers can destroy/omit shape-key data.
#
# Output:
#   ~/Desktop/MotionStage/data/unreal/motionstage_mixamo_final.fbx
#
# Manifest:
#   ~/Desktop/MotionStage/data/unreal/motionstage_mixamo_final_manifest.json
#
# Run INSIDE Blender with the completed Stage 19E scene open.

import bpy
import json
from pathlib import Path


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

PROJECT_ROOT = (
    Path.home()
    / "Desktop"
    / "MotionStage"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "unreal"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "motionstage_mixamo_final.fbx"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "motionstage_mixamo_final_manifest.json"
)

FINAL_SURFACE_NAME = (
    "Alpha_Surface_MS19E7D_KneeCalibrated"
)

OPTIONAL_JOINTS_NAME = "Alpha_Joints"

TARGET_REQUIRED_BONE = "mixamorig:Hips"

EXPECTED_ARMATURE_ACTION_PREFIX = (
    "MotionStage_Retargeted_Mixamo"
)

EXPECTED_SHAPE_ACTION = (
    "MotionStage_19E7D_KneeCorrective_ShapeKeys"
)

EXPECTED_LEFT_SHAPE_KEY = (
    "MS19E7D_LeftKneeCorrective"
)

EXPECTED_RIGHT_SHAPE_KEY = (
    "MS19E7D_RightKneeCorrective"
)


# ---------------------------------------------------------------------
# HELPERS
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


def uses_armature(mesh_obj, armature_obj):
    if mesh_obj is None or mesh_obj.type != "MESH":
        return False

    for modifier in mesh_obj.modifiers:
        if (
            modifier.type == "ARMATURE"
            and modifier.object == armature_obj
        ):
            return True

    current = mesh_obj.parent

    while current is not None:
        if current == armature_obj:
            return True
        current = current.parent

    return False


def get_active_action(obj):
    if (
        obj.animation_data is None
        or obj.animation_data.action is None
    ):
        return None

    return obj.animation_data.action


def get_shape_action(mesh_obj):
    shape_keys = mesh_obj.data.shape_keys

    if shape_keys is None:
        return None

    if (
        shape_keys.animation_data is None
        or shape_keys.animation_data.action is None
    ):
        return None

    return shape_keys.animation_data.action


def shape_key_names(mesh_obj):
    shape_keys = mesh_obj.data.shape_keys

    if shape_keys is None:
        return []

    return [
        key.name
        for key in shape_keys.key_blocks
    ]


def save_selection_state():
    return {
        "selected": [
            obj.name
            for obj in bpy.context.selected_objects
        ],
        "active": (
            bpy.context.view_layer.objects.active.name
            if bpy.context.view_layer.objects.active
            else None
        ),
        "frame": int(bpy.context.scene.frame_current),
    }


def restore_selection_state(state):
    bpy.ops.object.select_all(
        action="DESELECT"
    )

    for name in state["selected"]:
        obj = bpy.data.objects.get(name)

        if obj is not None:
            obj.select_set(True)

    if state["active"]:
        active = bpy.data.objects.get(
            state["active"]
        )

        if active is not None:
            bpy.context.view_layer.objects.active = active

    bpy.context.scene.frame_set(
        state["frame"]
    )


# ---------------------------------------------------------------------
# VALIDATE FINAL STAGE 19E ASSETS
# ---------------------------------------------------------------------

armature = find_mixamo_armature()

final_surface = bpy.data.objects.get(
    FINAL_SURFACE_NAME
)

if final_surface is None:
    raise RuntimeError(
        "Final Stage 19E mesh was not found:\n"
        f"  {FINAL_SURFACE_NAME}\n\n"
        "Do not export an earlier calibration mesh."
    )

if final_surface.type != "MESH":
    raise RuntimeError(
        f"'{FINAL_SURFACE_NAME}' exists but is not a mesh."
    )

if not uses_armature(
    final_surface,
    armature,
):
    raise RuntimeError(
        f"'{FINAL_SURFACE_NAME}' is not skinned to "
        f"Mixamo armature '{armature.name}'."
    )

armature_action = get_active_action(
    armature
)

if armature_action is None:
    raise RuntimeError(
        "The Mixamo armature has no active animation action."
    )

if not armature_action.name.startswith(
    EXPECTED_ARMATURE_ACTION_PREFIX
):
    raise RuntimeError(
        "Unexpected Mixamo action. Expected a MotionStage "
        "retargeted action, found:\n"
        f"  {armature_action.name}"
    )

keys = shape_key_names(
    final_surface
)

for required_key in (
    EXPECTED_LEFT_SHAPE_KEY,
    EXPECTED_RIGHT_SHAPE_KEY,
):
    if required_key not in keys:
        raise RuntimeError(
            "Final Stage 19E mesh is missing required shape key:\n"
            f"  {required_key}"
        )

shape_action = get_shape_action(
    final_surface
)

if shape_action is None:
    raise RuntimeError(
        "The final Stage 19E mesh has the corrective shape keys "
        "but no active shape-key animation action."
    )

if shape_action.name != EXPECTED_SHAPE_ACTION:
    print(
        "WARNING: expected shape-key action "
        f"'{EXPECTED_SHAPE_ACTION}', but found "
        f"'{shape_action.name}'."
    )


# ---------------------------------------------------------------------
# BUILD CLEAN EXPORT SELECTION
# ---------------------------------------------------------------------

objects_to_export = [
    armature,
    final_surface,
]

joints_mesh = bpy.data.objects.get(
    OPTIONAL_JOINTS_NAME
)

if (
    joints_mesh is not None
    and joints_mesh.type == "MESH"
    and uses_armature(
        joints_mesh,
        armature,
    )
):
    objects_to_export.append(
        joints_mesh
    )

selection_state = save_selection_state()

visibility_state = {
    obj.name: {
        "hide": obj.hide_get(),
        "hide_viewport": obj.hide_viewport,
    }
    for obj in objects_to_export
}

frame_start = int(
    bpy.context.scene.frame_start
)

frame_end = int(
    bpy.context.scene.frame_end
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

try:
    bpy.ops.object.select_all(
        action="DESELECT"
    )

    for obj in objects_to_export:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)

    bpy.context.view_layer.objects.active = (
        armature
    )

    bpy.context.scene.frame_set(
        frame_start
    )

    bpy.context.view_layer.update()

    # use_mesh_modifiers=False is intentional because the final knee
    # correction lives in animated shape keys.
    bpy.ops.export_scene.fbx(
        filepath=str(
            OUTPUT_PATH
        ),

        use_selection=True,

        object_types={
            "ARMATURE",
            "MESH",
        },

        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",

        use_mesh_modifiers=False,

        add_leaf_bones=False,
        use_armature_deform_only=False,

        primary_bone_axis="Y",
        secondary_bone_axis="X",

        axis_forward="-Z",
        axis_up="Y",

        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,

        path_mode="AUTO",
    )

finally:
    for obj in objects_to_export:
        state = visibility_state[obj.name]
        obj.hide_set(
            state["hide"]
        )
        obj.hide_viewport = (
            state["hide_viewport"]
        )

    restore_selection_state(
        selection_state
    )


# ---------------------------------------------------------------------
# VERIFY OUTPUT + WRITE MANIFEST
# ---------------------------------------------------------------------

if not OUTPUT_PATH.exists():
    raise RuntimeError(
        "FBX export command completed, but the output file "
        "was not found:\n"
        f"  {OUTPUT_PATH}"
    )

size_bytes = OUTPUT_PATH.stat().st_size
size_mb = size_bytes / (1024 * 1024)

manifest = {
    "stage": "19F_final_unreal_fbx_export",
    "fbx_path": str(OUTPUT_PATH),
    "fbx_size_bytes": int(size_bytes),

    "mixamo_armature": armature.name,
    "armature_action": armature_action.name,

    "final_surface": final_surface.name,
    "shape_key_action": shape_action.name,
    "corrective_shape_keys": [
        EXPECTED_LEFT_SHAPE_KEY,
        EXPECTED_RIGHT_SHAPE_KEY,
    ],

    "optional_joints_mesh": (
        joints_mesh.name
        if joints_mesh in objects_to_export
        else None
    ),

    "exported_objects": [
        obj.name
        for obj in objects_to_export
    ],

    "frame_range": [
        frame_start,
        frame_end,
    ],

    "export_settings": {
        "use_selection": True,
        "use_mesh_modifiers": False,
        "add_leaf_bones": False,
        "bake_anim": True,
        "bake_anim_use_all_bones": True,
        "bake_anim_use_all_actions": False,
        "bake_anim_simplify_factor": 0.0,
        "axis_forward": "-Z",
        "axis_up": "Y",
    },

    "notes": [
        "Only the validated final Stage 19E surface is exported.",
        "Earlier MotionStage diagnostic/calibration duplicate meshes are excluded.",
        "Mesh modifiers are not applied so corrective shape keys remain exportable.",
        "Enable Import Morph Targets when importing the FBX into Unreal Engine.",
    ],
}

with MANIFEST_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
    )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

print()
print("=" * 88)
print("MotionStage Stage 19F - FINAL UNREAL FBX EXPORT")
print("=" * 88)

print(
    "Mixamo armature       :",
    armature.name,
)

print(
    "Armature action       :",
    armature_action.name,
)

print(
    "Final surface         :",
    final_surface.name,
)

print(
    "Shape-key action      :",
    shape_action.name,
)

print(
    "Corrective shape keys :",
    EXPECTED_LEFT_SHAPE_KEY,
    ",",
    EXPECTED_RIGHT_SHAPE_KEY,
)

print(
    "Additional joint mesh :",
    (
        joints_mesh.name
        if joints_mesh in objects_to_export
        else "not exported / not present"
    ),
)

print(
    "Frames                :",
    f"{frame_start} -> {frame_end}",
)

print(
    "FBX                   :",
    OUTPUT_PATH,
)

print(
    "FBX size              :",
    f"{size_mb:.2f} MB",
)

print(
    "Manifest              :",
    MANIFEST_PATH,
)

print()
print(
    "Stage 19F export completed without modifying the "
    "Stage 19E rig, animation, mesh, shape keys, or grounding."
)

print()
print(
    "Next: import the FBX into Unreal Engine with "
    "'Import Morph Targets' enabled."
)

print("=" * 88)
