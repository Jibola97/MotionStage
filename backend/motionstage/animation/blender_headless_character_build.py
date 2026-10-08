# MotionStage Stage 22B — Headless Blender Character Builder
#
# Run by Blender in background mode. It:
#   1) reads a Stage 19A animation_package.json,
#   2) builds the MotionStage_RetargetRig,
#   3) imports the clean Mixamo Y-Bot FBX,
#   4) applies the proven Stage 21F v4 pose-basis retarget,
#   5) validates the result numerically,
#   6) saves a .blend snapshot,
#   7) exports an animated FBX.
#
# Blender invocation example:
#   blender -b --factory-startup \
#     --python blender_headless_character_build.py -- \
#     --project-root ~/Desktop/MotionStage \
#     --performance test_performance_02 \
#     --ybot ~/Desktop/MotionStage/assets/characters/ybot.fbx \
#     --output-dir ~/Desktop/MotionStage/data/animation/test_performance_02/character

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--performance", required=True)
    parser.add_argument("--ybot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


ARGS = parse_args()
PROJECT_ROOT = Path(ARGS.project_root).expanduser().resolve()
PERFORMANCE = ARGS.performance
YBOT_PATH = Path(ARGS.ybot).expanduser().resolve()
OUTPUT_DIR = Path(ARGS.output_dir).expanduser().resolve()

PACKAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "animation"
    / PERFORMANCE
    / "animation_package.json"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "backend"
    / "motionstage"
    / "animation"
    / "motionstage_retarget_map.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FBX_PATH = OUTPUT_DIR / f"{PERFORMANCE}_ybot.fbx"
BLEND_PATH = OUTPUT_DIR / f"{PERFORMANCE}_ybot.blend"
MANIFEST_PATH = OUTPUT_DIR / "character_build_manifest.json"

SOURCE_RIG_NAME = "MotionStage_RetargetRig"
FINAL_ARMATURE_NAME = "MotionStage_FinalArmature"
FINAL_ACTION_NAME = "MotionStage_Final_Action"

ROOT_BONE_LENGTH = 0.08
HAND_EXTENSION_RATIO = 0.28

# Production output should be centred rather than visually offset.
SOURCE_DISPLAY_OFFSET_X = 0.0
FINAL_OFFSET_X = 0.0

BONE_MAP = [
    ("pelvis",       "mixamorig:Hips"),
    ("spine_01",     "mixamorig:Spine"),
    ("spine_02",     "mixamorig:Spine2"),
    ("neck_01",      "mixamorig:Neck"),
    ("head",         "mixamorig:Head"),

    ("clavicle_l",   "mixamorig:LeftShoulder"),
    ("upperarm_l",   "mixamorig:LeftArm"),
    ("lowerarm_l",   "mixamorig:LeftForeArm"),
    ("hand_l",       "mixamorig:LeftHand"),

    ("clavicle_r",   "mixamorig:RightShoulder"),
    ("upperarm_r",   "mixamorig:RightArm"),
    ("lowerarm_r",   "mixamorig:RightForeArm"),
    ("hand_r",       "mixamorig:RightHand"),

    ("thigh_l",      "mixamorig:LeftUpLeg"),
    ("calf_l",       "mixamorig:LeftLeg"),
    ("foot_l",       "mixamorig:LeftFoot"),

    ("thigh_r",      "mixamorig:RightUpLeg"),
    ("calf_r",       "mixamorig:RightLeg"),
    ("foot_r",       "mixamorig:RightFoot"),
]


# ---------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_vector(values):
    return Vector((float(values[0]), float(values[1]), float(values[2])))


def lerp(a, b, t):
    return a + ((b - a) * t)


def safe_direction(head, tail):
    direction = tail - head
    if direction.length < 1e-7:
        return Vector((0.0, 0.0, 0.05))
    return direction


def reset_scene():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove orphan object data created by factory scene / retries.
    for datablocks in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.actions,
        bpy.data.materials,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


# ---------------------------------------------------------------------
# STAGE 19B BRIDGE RIG
# ---------------------------------------------------------------------

def build_bridge_points(joints):
    pelvis = as_vector(joints["pelvis"])
    chest = as_vector(joints["chest"])
    head = as_vector(joints["head"])

    left_shoulder = as_vector(joints["left_shoulder"])
    right_shoulder = as_vector(joints["right_shoulder"])
    left_elbow = as_vector(joints["left_elbow"])
    right_elbow = as_vector(joints["right_elbow"])
    left_wrist = as_vector(joints["left_wrist"])
    right_wrist = as_vector(joints["right_wrist"])

    left_hip = as_vector(joints["left_hip"])
    right_hip = as_vector(joints["right_hip"])
    left_knee = as_vector(joints["left_knee"])
    right_knee = as_vector(joints["right_knee"])
    left_ankle = as_vector(joints["left_ankle"])
    right_ankle = as_vector(joints["right_ankle"])
    left_foot = as_vector(joints["left_foot"])
    right_foot = as_vector(joints["right_foot"])

    spine_01_point = lerp(pelvis, chest, 0.34)
    spine_02_point = lerp(pelvis, chest, 0.67)
    neck = lerp(chest, head, 0.38)

    left_hand = left_wrist + (left_wrist - left_elbow) * HAND_EXTENSION_RATIO
    right_hand = right_wrist + (right_wrist - right_elbow) * HAND_EXTENSION_RATIO

    return {
        "root_head": pelvis,
        "root_tail": pelvis + Vector((0.0, 0.0, ROOT_BONE_LENGTH)),
        "pelvis": pelvis,
        "spine_01_point": spine_01_point,
        "spine_02_point": spine_02_point,
        "chest": chest,
        "neck": neck,
        "head": head,
        "left_shoulder": left_shoulder,
        "left_elbow": left_elbow,
        "left_wrist": left_wrist,
        "left_hand": left_hand,
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
        "right_hand": right_hand,
        "left_hip": left_hip,
        "left_knee": left_knee,
        "left_ankle": left_ankle,
        "left_foot": left_foot,
        "right_hip": right_hip,
        "right_knee": right_knee,
        "right_ankle": right_ankle,
        "right_foot": right_foot,
    }


def bridge_bone_geometry(points):
    return {
        "root": (points["root_head"], points["root_tail"]),
        "pelvis": (points["pelvis"], points["spine_01_point"]),
        "spine_01": (points["spine_01_point"], points["spine_02_point"]),
        "spine_02": (points["spine_02_point"], points["chest"]),
        "neck_01": (points["chest"], points["neck"]),
        "head": (points["neck"], points["head"]),

        "clavicle_l": (points["chest"], points["left_shoulder"]),
        "upperarm_l": (points["left_shoulder"], points["left_elbow"]),
        "lowerarm_l": (points["left_elbow"], points["left_wrist"]),
        "hand_l": (points["left_wrist"], points["left_hand"]),

        "clavicle_r": (points["chest"], points["right_shoulder"]),
        "upperarm_r": (points["right_shoulder"], points["right_elbow"]),
        "lowerarm_r": (points["right_elbow"], points["right_wrist"]),
        "hand_r": (points["right_wrist"], points["right_hand"]),

        "thigh_l": (points["left_hip"], points["left_knee"]),
        "calf_l": (points["left_knee"], points["left_ankle"]),
        "foot_l": (points["left_ankle"], points["left_foot"]),

        "thigh_r": (points["right_hip"], points["right_knee"]),
        "calf_r": (points["right_knee"], points["right_ankle"]),
        "foot_r": (points["right_ankle"], points["right_foot"]),
    }


def desired_bone_matrix(head, tail, rest_length):
    direction = safe_direction(head, tail)
    target_length = direction.length

    rotation = (
        direction.normalized()
        .to_track_quat("Y", "Z")
        .to_matrix()
        .to_4x4()
    )

    scale_y = target_length / rest_length if rest_length > 1e-8 else 1.0
    scale = Matrix.Diagonal((1.0, scale_y, 1.0, 1.0))

    return Matrix.Translation(head) @ rotation @ scale


def mapping_hierarchy(mapping):
    hierarchy = mapping.get("bridge_hierarchy", [])
    if not hierarchy:
        raise ValueError("Retarget mapping has no bridge_hierarchy.")
    return hierarchy


def ordered_bone_names(mapping):
    hierarchy = mapping_hierarchy(mapping)
    parent_lookup = {item["name"]: item.get("parent") for item in hierarchy}

    remaining = set(parent_lookup)
    ordered = []

    while remaining:
        progress = False
        for name in list(remaining):
            parent = parent_lookup[name]
            if parent is None or parent in ordered or parent not in parent_lookup:
                ordered.append(name)
                remaining.remove(name)
                progress = True

        if not progress:
            ordered.extend(sorted(remaining))
            break

    return ordered


def build_bridge_rig(package, mapping):
    first_frame = package["frames"][0]
    points = build_bridge_points(first_frame["joints"])
    geometry = bridge_bone_geometry(points)
    hierarchy = mapping_hierarchy(mapping)

    required_names = {item["name"] for item in hierarchy}
    missing = required_names - set(geometry)
    if missing:
        raise RuntimeError(f"Missing bridge geometry: {sorted(missing)}")

    arm_data = bpy.data.armatures.new(SOURCE_RIG_NAME)
    arm_obj = bpy.data.objects.new(SOURCE_RIG_NAME, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    arm_obj.location.x = SOURCE_DISPLAY_OFFSET_X

    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = {}

    for item in hierarchy:
        name = item["name"]
        head, tail = geometry[name]

        bone = arm_data.edit_bones.new(name)
        bone.head = head
        bone.tail = head + safe_direction(head, tail)
        edit_bones[name] = bone

    for item in hierarchy:
        parent_name = item.get("parent")
        if not parent_name:
            continue
        child = edit_bones[item["name"]]
        child.parent = edit_bones[parent_name]
        child.use_connect = False

    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.ops.object.mode_set(mode="OBJECT")

    order = ordered_bone_names(mapping)
    rest_lengths = {
        name: max(float(arm_obj.data.bones[name].length), 1e-8)
        for name in order
    }

    arm_obj.animation_data_create()
    action = bpy.data.actions.new("MotionStage_Bridge_Action")
    arm_obj.animation_data.action = action

    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    for blender_frame, source in enumerate(package["frames"], start=1):
        bpy.context.scene.frame_set(blender_frame)

        points = build_bridge_points(source["joints"])
        geometry = bridge_bone_geometry(points)

        for name in order:
            head, tail = geometry[name]
            pb = arm_obj.pose.bones[name]

            pb.matrix = desired_bone_matrix(
                head=head,
                tail=tail,
                rest_length=rest_lengths[name],
            )

            pb.keyframe_insert("location", frame=blender_frame)
            pb.keyframe_insert("rotation_quaternion", frame=blender_frame)
            pb.keyframe_insert("scale", frame=blender_frame)

        bpy.context.view_layer.update()

    bpy.ops.object.mode_set(mode="OBJECT")
    arm_obj.show_in_front = True

    return arm_obj


# ---------------------------------------------------------------------
# MIXAMO IMPORT
# ---------------------------------------------------------------------

def import_ybot(path):
    if not path.exists():
        raise FileNotFoundError(f"Y-Bot FBX not found: {path}")

    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]

    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(
            f"Expected exactly one imported armature, found {len(armatures)}: "
            f"{[o.name for o in armatures]}"
        )

    arm = armatures[0]

    meshes = [
        obj
        for obj in imported
        if obj.type == "MESH"
        and any(
            mod.type == "ARMATURE" and mod.object == arm
            for mod in obj.modifiers
        )
    ]

    if not meshes:
        raise RuntimeError("Imported Y-Bot has no skinned meshes.")

    return arm, meshes


# ---------------------------------------------------------------------
# STAGE 21F v4 RETARGET
# ---------------------------------------------------------------------

def hierarchy_order(armature_obj):
    ordered = []

    def walk(bone):
        ordered.append(bone.name)
        for child in bone.children:
            walk(child)

    for root in [b for b in armature_obj.data.bones if b.parent is None]:
        walk(root)

    return ordered


def pose_world_direction(arm_obj, pose_bone):
    vec_obj = pose_bone.tail - pose_bone.head
    vec_world = arm_obj.matrix_world.to_3x3() @ vec_obj

    if vec_world.length < 1e-10:
        raise RuntimeError(f"Zero-length pose bone: {pose_bone.name}")

    return vec_world.normalized()


def rest_world_direction(arm_obj, data_bone):
    vec_obj = data_bone.tail_local - data_bone.head_local
    vec_world = arm_obj.matrix_world.to_3x3() @ vec_obj

    if vec_world.length < 1e-10:
        raise RuntimeError(f"Zero-length rest bone: {data_bone.name}")

    return vec_world.normalized()


def replace_rotation(matrix, quat):
    out = quat.to_matrix().to_4x4()
    out.translation = matrix.translation.copy()
    return out


def build_final_character(source_rig, original_ybot, original_meshes, frame_count):
    final_arm = original_ybot.copy()
    final_arm.data = original_ybot.data.copy()
    final_arm.name = FINAL_ARMATURE_NAME
    final_arm.data.name = FINAL_ARMATURE_NAME + "_Data"
    bpy.context.scene.collection.objects.link(final_arm)

    final_arm.matrix_world = original_ybot.matrix_world.copy()
    final_arm.location.x = FINAL_OFFSET_X
    final_arm.data.pose_position = "POSE"
    final_arm.animation_data_clear()

    for pb in final_arm.pose.bones:
        for constraint in list(pb.constraints):
            pb.constraints.remove(constraint)
        pb.matrix_basis.identity()
        pb.rotation_mode = "QUATERNION"

    final_meshes = []

    for obj in original_meshes:
        dup = obj.copy()
        dup.data = obj.data.copy()
        dup.name = "MotionStage_Final_" + obj.name
        dup.data.name = dup.name + "_Mesh"
        bpy.context.scene.collection.objects.link(dup)

        dup.parent = final_arm
        dup.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
        dup.matrix_basis = obj.matrix_basis.copy()
        dup.animation_data_clear()

        for mod in dup.modifiers:
            if mod.type == "ARMATURE" and mod.object == original_ybot:
                mod.object = final_arm

        final_meshes.append(dup)

    for source_name, target_name in BONE_MAP:
        if source_name not in source_rig.data.bones:
            raise RuntimeError(f"Missing source bone: {source_name}")
        if target_name not in final_arm.data.bones:
            raise RuntimeError(f"Missing target Mixamo bone: {target_name}")

    target_order = hierarchy_order(final_arm)
    source_for_target = {target: source for source, target in BONE_MAP}
    mapped_targets = set(source_for_target)

    target_rest_world_quat = {}
    target_rest_world_dir = {}

    for target_name in mapped_targets:
        bone = final_arm.data.bones[target_name]
        rest_world = final_arm.matrix_world @ bone.matrix_local

        target_rest_world_quat[target_name] = (
            rest_world.to_quaternion().normalized()
        )
        target_rest_world_dir[target_name] = rest_world_direction(final_arm, bone)

    final_arm.animation_data_create()
    action = bpy.data.actions.new(FINAL_ACTION_NAME)
    final_arm.animation_data.action = action

    for frame in range(1, frame_count + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        desired_pose = {}

        for bone_name in target_order:
            bone = final_arm.data.bones[bone_name]

            if bone.parent is None:
                base = bone.matrix_local.copy()
            else:
                parent_pose = desired_pose[bone.parent.name]
                rest_relative = (
                    bone.parent.matrix_local.inverted() @ bone.matrix_local
                )
                base = parent_pose @ rest_relative

            if bone_name in mapped_targets:
                source_name = source_for_target[bone_name]
                source_dir_world = pose_world_direction(
                    source_rig,
                    source_rig.pose.bones[source_name],
                )

                align_world = target_rest_world_dir[
                    bone_name
                ].rotation_difference(source_dir_world)

                desired_world_quat = (
                    align_world @ target_rest_world_quat[bone_name]
                ).normalized()

                final_world_quat = (
                    final_arm.matrix_world.to_quaternion().normalized()
                )

                desired_object_quat = (
                    final_world_quat.inverted() @ desired_world_quat
                ).normalized()

                base = replace_rotation(base, desired_object_quat)

            desired_pose[bone_name] = base

        for bone_name in target_order:
            bone = final_arm.data.bones[bone_name]
            pb = final_arm.pose.bones[bone_name]

            if bone.parent is None:
                basis = bone.convert_local_to_pose(
                    desired_pose[bone_name],
                    bone.matrix_local,
                    invert=True,
                )
            else:
                basis = bone.convert_local_to_pose(
                    desired_pose[bone_name],
                    bone.matrix_local,
                    parent_matrix=desired_pose[bone.parent.name],
                    parent_matrix_local=bone.parent.matrix_local,
                    invert=True,
                )

            loc, rot, scale = basis.decompose()
            pb.location = loc
            pb.rotation_quaternion = rot.normalized()
            pb.scale = scale

            if bone_name in mapped_targets:
                pb.keyframe_insert("location", frame=frame, group=bone_name)
                pb.keyframe_insert(
                    "rotation_quaternion",
                    frame=frame,
                    group=bone_name,
                )
                pb.keyframe_insert("scale", frame=frame, group=bone_name)

        if (frame - 1) % 30 == 0:
            print(f"[22B] retargeted frame {frame}/{frame_count}")

    for fcurve in action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"

    return final_arm, final_meshes, action


def validate_pose(source_rig, final_arm, frame_count):
    frames = sorted(
        set(
            [
                1,
                1 + (frame_count - 1) // 4,
                1 + (frame_count - 1) // 2,
                1 + 3 * (frame_count - 1) // 4,
                frame_count,
            ]
        )
    )

    results = []

    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        errors = []

        for source_name, target_name in BONE_MAP:
            source_dir = pose_world_direction(
                source_rig,
                source_rig.pose.bones[source_name],
            )
            target_dir = pose_world_direction(
                final_arm,
                final_arm.pose.bones[target_name],
            )

            dot = max(-1.0, min(1.0, source_dir.dot(target_dir)))
            errors.append(math.degrees(math.acos(dot)))

        result = {
            "frame": frame,
            "mean_error_degrees": sum(errors) / len(errors),
            "max_error_degrees": max(errors),
        }
        results.append(result)

        print(
            "[22B] validation "
            f"frame={frame} "
            f"mean={result['mean_error_degrees']:.3f} "
            f"max={result['max_error_degrees']:.3f}"
        )

    return results


# ---------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------

def export_outputs(final_arm, final_meshes):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(PACKAGE["frames"])
    scene.frame_set(1)

    # Save debug/reproducibility snapshot first.
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))

    bpy.ops.object.select_all(action="DESELECT")
    final_arm.select_set(True)
    for obj in final_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = final_arm

    bpy.ops.export_scene.fbx(
        filepath=str(FBX_PATH),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=0.0,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

print("=" * 72)
print("MotionStage Stage 22B — Headless Blender Character Build")
print("=" * 72)
print("Performance :", PERFORMANCE)
print("Package     :", PACKAGE_PATH)
print("Y-Bot       :", YBOT_PATH)
print("Output dir  :", OUTPUT_DIR)

PACKAGE = load_json(PACKAGE_PATH)
MAPPING = load_json(MAPPING_PATH)

if MAPPING.get("schema") != "motionstage.retarget_map.v1":
    raise RuntimeError(
        "Unexpected retarget-map schema: "
        f"{MAPPING.get('schema')!r}"
    )

if not PACKAGE.get("frames"):
    raise RuntimeError("Animation package contains no frames.")

reset_scene()

source_rig = build_bridge_rig(PACKAGE, MAPPING)
original_ybot, original_meshes = import_ybot(YBOT_PATH)

frame_count = len(PACKAGE["frames"])

final_arm, final_meshes, final_action = build_final_character(
    source_rig=source_rig,
    original_ybot=original_ybot,
    original_meshes=original_meshes,
    frame_count=frame_count,
)

validation = validate_pose(
    source_rig=source_rig,
    final_arm=final_arm,
    frame_count=frame_count,
)

# Fail automation if the proven retarget suddenly regresses.
worst_mean = max(item["mean_error_degrees"] for item in validation)
worst_max = max(item["max_error_degrees"] for item in validation)

if worst_mean > 1.0 or worst_max > 5.0:
    raise RuntimeError(
        "Retarget validation failed: "
        f"worst mean={worst_mean:.3f}°, "
        f"worst max={worst_max:.3f}°"
    )

export_outputs(final_arm, final_meshes)

manifest = {
    "stage": "22B",
    "status": "completed",
    "performance_name": PERFORMANCE,
    "frame_count": frame_count,
    "root_policy": "in_place_pelvis_relative",
    "source_animation_package": str(PACKAGE_PATH),
    "ybot_asset": str(YBOT_PATH),
    "blend_output": str(BLEND_PATH),
    "fbx_output": str(FBX_PATH),
    "validation": validation,
    "validation_thresholds": {
        "max_allowed_mean_error_degrees": 1.0,
        "max_allowed_peak_error_degrees": 5.0,
    },
}

MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print()
print("=" * 72)
print("STAGE 22B BLENDER PASS")
print("=" * 72)
print("FBX      :", FBX_PATH)
print("BLEND    :", BLEND_PATH)
print("Manifest :", MANIFEST_PATH)
print(
    "Worst validation: "
    f"mean={worst_mean:.3f}°, max={worst_max:.3f}°"
)
print("=" * 72)
