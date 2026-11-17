"""Convert the official URDF into a local MuJoCo development scene."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from piper_sim.calibration import DEFAULT_CONFIG, R_OPTICAL_MUJOCO, load_config, native_path

ROOT = Path(native_path(Path(__file__).resolve().parents[1]))
SOURCE = ROOT / "references/upstream/piper_ros/src/piper_description"
OUTPUT = ROOT / "assets/piper"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(SOURCE / "urdf/piper_description.urdf").getroot()
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(SOURCE / mesh.get("filename").split("piper_description/")[1]))
    ext = ET.SubElement(robot, "mujoco")
    ET.SubElement(ext, "compiler", discardvisual="false", fusestatic="false")
    urdf = OUTPUT / "resolved.urdf"
    ET.ElementTree(robot).write(urdf)
    model = mujoco.MjModel.from_xml_path(str(urdf))
    output = OUTPUT / "scene.xml"
    mujoco.mj_saveLastXML(str(output), model)
    scene = ET.parse(output)
    root = scene.getroot()
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="1", armature="0.01")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .1", rgba=".2 .25 .3 1")
    grip = next(b for b in root.iter("body") if b.get("name") == "gripper_base")
    ET.SubElement(grip, "site", name="tcp", pos="0 0 .12", size=".008", rgba="1 0 0 1")
    # Pose is supplied in the optical convention; convert only at the rendering boundary.
    hand_eye = cfg["hand_eye"]
    parent = next((b for b in root.iter("body") if b.get("name") == hand_eye["parent_body"]), None)
    if parent is None:
        raise ValueError(f"Unknown URDF parent_body: {hand_eye['parent_body']}")
    t = np.asarray(hand_eye["T_end_camera"])
    xyzw = Rotation.from_matrix(t[:3, :3] @ R_OPTICAL_MUJOCO).as_quat()
    pos = " ".join(map(str, t[:3, 3]))
    quat = " ".join(map(str, xyzw[[3, 0, 1, 2]]))
    cam = cfg["camera"]
    k = cam["intrinsics"]
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(cam["width"]), offheight=str(cam["height"]))
    ET.SubElement(
        parent,
        "camera",
        name=cam["name"],
        mode="fixed",
        pos=pos,
        quat=quat,
        resolution=f"{cam['width']} {cam['height']}",
        sensorsize="1 1",
        focalpixel=f"{k['fx']} {k['fy']}",
        principalpixel=f"{cam['width'] / 2 - k['cx']} {cam['height'] / 2 - k['cy']}",
    )
    if cfg["simulation"]["show_camera_marker"]:
        marker = ET.SubElement(parent, "body", name="camera_visual_marker", pos=pos, quat=quat)
        # The optical origin lies on the front face; housing extends behind the lenses.
        visual_cfg = cfg["simulation"]["camera_visual"]
        width, height, depth = visual_cfg["housing_size_m"]
        no_collision = {"contype": "0", "conaffinity": "0", "mass": "0", "group": "3"}
        ET.SubElement(
            marker,
            "geom",
            name="camera_housing",
            type="box",
            size=f"{width / 2} {height / 2} {depth / 2}",
            pos=f"0 0 {depth / 2}",
            rgba=".85 .87 .88 1",
            **no_collision,
        )
        for index, x in enumerate([-width * 0.35, 0, width * 0.35]):
            ET.SubElement(
                marker,
                "geom",
                name=f"camera_lens_{index}",
                type="cylinder",
                size=".005 .001",
                pos=f"{x} 0 -.001",
                rgba=".02 .03 .04 1",
                **no_collision,
            )
        ET.SubElement(
            parent,
            "geom",
            name="camera_bracket",
            type="box",
            size=" ".join(str(x / 2) for x in visual_cfg["bracket_size_m"]),
            pos=" ".join(map(str, visual_cfg["bracket_position_m"])),
            rgba=".12 .13 .15 1",
            **no_collision,
        )
    if cfg["simulation"]["demo_targets"]:
        ET.SubElement(
            world,
            "geom",
            name="demo_table",
            type="box",
            pos=".32 0 .04",
            size=".28 .25 .04",
            rgba=".65 .65 .6 1",
        )
        for name, xyz, color in [
            ("red", ".389 0 .105", ".9 .05 .05 1"),
            ("green", ".44 .065 .105", ".05 .8 .1 1"),
            ("blue", ".34 -.065 .105", ".05 .15 .95 1"),
        ]:
            ET.SubElement(
                world,
                "geom",
                name=f"target_{name}",
                type="box",
                pos=xyz,
                size=".025 .025 .025",
                rgba=color,
            )
    # Adjacent CAD meshes overlap around physical joint housings. The welded base
    # is treated as world by MuJoCo, so default parent filtering does not exclude it.
    contact = ET.SubElement(root, "contact")
    for joint in robot.findall("joint"):
        ET.SubElement(
            contact,
            "exclude",
            body1=joint.find("parent").get("link"),
            body2=joint.find("child").get("link"),
        )
    actuator = ET.SubElement(root, "actuator")
    for j in range(1, 9):
        name = f"joint{j}"
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[idx]
        # Development gains, not measured real hardware parameters.
        ET.SubElement(
            actuator,
            "position",
            name=f"pos_{name}",
            joint=name,
            kp="100" if j <= 6 else "200",
            kv="10" if j <= 6 else "2",
            ctrllimited="true",
            ctrlrange=f"{low} {high}",
            forcelimited="true",
            forcerange="-50 50" if j <= 6 else "-10 10",
        )
    ET.indent(root)
    # User measured competition gripper: total opening 100 mm.
    for name,limits in [('joint7','0 0.05'),('joint8','-0.05 0')]:
        scene.getroot().find(".//joint[@name='"+name+"']").set('range',limits)
        scene.getroot().find(".//position[@joint='"+name+"']").set('ctrlrange',limits)
    scene.write(output)
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nq == 8 and model.nu == 8
    print(f"{output}: nq={model.nq}, nu={model.nu}, meshes={model.nmesh}")


if __name__ == "__main__":
    main()
