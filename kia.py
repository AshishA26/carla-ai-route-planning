"""Render the team's Kia Soul EV (with its sensor rig) over the CARLA ego vehicle.

CARLA's packaged release cannot add a new drivable vehicle, so a small CARLA car does the driving
and this module draws the Kia over it from the recorded camera. The Kia is larger than the car
underneath, so it hides it.

Model: "Kia Soul" by Nieve5677 on Sketchfab, CC BY 4.0
(https://sketchfab.com/3d-models/kia-soul-079c8678206c47a087d0ae47bcfe8055). Changes: repainted white
with a black roof, sensor rig added, wheels separated so they turn.
Rig dimensions: src/interfacing/eve_description/urdf in the WATonomous monorepo.
"""
import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh

if not hasattr(np, "infty"):  # pyrender 0.1.45 still uses the NumPy 1 alias
    np.infty = np.inf

HERE = os.path.dirname(os.path.abspath(__file__))
GLB = os.path.join(HERE, "kia-model", "kia_soul.glb")
CREDIT = "Kia Soul model by Nieve5677 (Sketchfab), CC BY 4.0, repainted"

# Eve (URDF): wheel radius, wheelbase, mast position, sensor heights
WHEELBASE = 2.5667
WHEEL_R = 0.31235
MAST_FROM_REAR_AXLE = 0.8629
ROOF_Z = 1.76          # base_link height (roof frame)
LIDAR_CC_Z = ROOF_Z + 0.35
CAM_RING_Z = ROOF_Z + 0.217
CAM_RING_R = 0.240676
MIRROR_LIDARS = [(1.02, 0.86, 1.29), (0.99, -0.90, 1.27)]  # (ahead of mast, right +, up), CARLA axes

PAINT = {  # material name -> (rgba, metallic, roughness)
    "body": ((0.93, 0.93, 0.91, 1.0), 0.15, 0.28),
    "roof": ((0.03, 0.03, 0.035, 1.0), 0.2, 0.25),
    # in this model "glass"/"r_glass"/"o_glass" are lamp lenses (headlights, tail lamps); windows use material_7
    "glass": ((0.92, 0.93, 0.95, 0.22), 0.0, 0.05),
    "r_glass": ((0.75, 0.08, 0.08, 0.30), 0.0, 0.05),
    "silver": ((0.62, 0.63, 0.65, 1.0), 0.8, 0.3),
    "chrome": ((0.8, 0.8, 0.82, 1.0), 0.9, 0.15),
    "material_7": ((0.03, 0.03, 0.03, 1.0), 0.0, 0.8),
    "black": ((0.02, 0.02, 0.02, 1.0), 0.0, 0.6),
    "black_m": ((0.02, 0.02, 0.02, 1.0), 0.0, 0.85),
    "interior": ((0.08, 0.08, 0.08, 1.0), 0.0, 0.9),
    "orange": ((0.9, 0.45, 0.1, 1.0), 0.0, 0.3),
    "o_glass": ((0.95, 0.5, 0.08, 0.35), 0.0, 0.1),
    "material": ((0.55, 0.03, 0.03, 1.0), 0.0, 0.3),  # tail lamps (given a dusk glow below)
}
RIG_BLACK = ((0.02, 0.02, 0.022, 1.0), 0.3, 0.45)
RIG_GREY = ((0.18, 0.18, 0.19, 1.0), 0.5, 0.35)
RIG_WHITE = ((0.9, 0.9, 0.88, 1.0), 0.0, 0.5)


GLOW = {"material": (0.45, 0.02, 0.02)}  # tail lamps on at dusk


def _mat(spec, key=None):
    import pyrender
    rgba, metal, rough = spec
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=rgba, metallicFactor=metal, roughnessFactor=rough,
                                              emissiveFactor=GLOW.get(key, (0.0, 0.0, 0.0)),
                                              alphaMode="BLEND" if rgba[3] < 1 else "OPAQUE", doubleSided=True)


def load_parts():
    """Returns ([(trimesh, paint_key, is_wheel, wheel_centre)], info) in CARLA vehicle axes:
    x forward, y right, z up, metres, origin on the ground midway between the axles."""
    sc = trimesh.load(GLB, force="scene")
    meshes = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname].copy()
        g.apply_transform(T)
        mat = getattr(getattr(g.visual, "material", None), "name", None) or "black"
        meshes.append((g, mat))
    allv = np.vstack([g.vertices for g, _ in meshes])
    lo, hi = allv.min(0), allv.max(0)
    # GLB: x = width, y = up, z = length (cm). Find the front: the tail lights are at the back.
    tail = np.vstack([g.vertices for g, m in meshes if m == "material"]) if any(m == "material" for _, m in meshes) else None
    front_sign = -1.0 if tail is not None and tail[:, 2].mean() > (lo[2] + hi[2]) / 2 else 1.0
    scale = 0.01

    def to_carla(v):
        x_fwd = front_sign * v[:, 2]
        y_right = -front_sign * v[:, 0]  # right-handed GLB with y up: x points left when z points forward
        z_up = v[:, 1] - lo[1]
        return np.c_[x_fwd, y_right, z_up] * scale

    conv = []
    for g, m in meshes:
        v = to_carla(g.vertices)
        conv.append((trimesh.Trimesh(v, g.faces, process=False), m))
    # wheels: small parts low down near the four corners
    allc = np.vstack([g.vertices for g, _ in conv])
    xmid = (allc[:, 0].min() + allc[:, 0].max()) / 2
    parts = []
    cands = []
    for g, m in conv:
        ext = g.extents
        c = g.bounds.mean(0)
        if ext.max() < 0.8 and c[2] < 0.5 and abs(c[1]) > 0.45 and m in ("silver", "material_7", "chrome", "black", "black_m"):
            cands.append((g, m, c))
        else:
            parts.append([g, m, False, None])
    # cluster wheel parts into four corners and centre the axles
    fronts = [c for _, _, c in cands if c[0] > xmid]
    rears = [c for _, _, c in cands if c[0] <= xmid]
    fx = np.median([c[0] for c in fronts]) if fronts else xmid + WHEELBASE / 2
    rx = np.median([c[0] for c in rears]) if rears else xmid - WHEELBASE / 2
    axle_mid = (fx + rx) / 2
    for g, m, c in cands:
        wx = fx if c[0] > xmid else rx
        centre = np.array([wx, np.sign(c[1]) * abs(c[1]), WHEEL_R])
        parts.append([g, m, True, centre])
    # recentre so x = 0 midway between the axles
    for p in parts:
        p[0] = p[0].copy()
        p[0].apply_translation([-axle_mid, 0, 0])
        if p[3] is not None:
            p[3] = p[3] - [axle_mid, 0, 0]
    # paint: body above the glass line is the black roof and pillars (like the real car)
    # glass line from the side windows only (headlight lenses share the glass material and sit low)
    side = []
    for p in parts:
        if p[1] in ("glass", "r_glass"):
            tri = p[0].vertices[p[0].faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            keep = (np.abs(n[:, 1]) > 0.7) & (tri.mean(1)[:, 2] > 0.9)
            side.append(tri[keep].reshape(-1, 3))
    side = np.vstack(side)
    belt = np.percentile(side[:, 2], 2) + 0.09
    glass_front, glass_rear = side[:, 0].max(), side[:, 0].min()
    body_v = np.vstack([p[0].vertices for p in parts if p[1] == "body"])
    roof_z = np.percentile(body_v[:, 2], 99.5)
    out = []
    for g, m, wheel, centre in parts:
        if m == "body":
            tri = g.vertices[g.faces]
            fc = tri.mean(1)
            nz = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nz = nz[:, 2] / (np.linalg.norm(nz, axis=1) + 1e-9)
            # black roof and pillars, as on the real car: above the window line, between the
            # windscreen base and the rear pillar, but not the bonnet (faces up, well below the roof)
            nvec = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nx = nvec[:, 0] / (np.linalg.norm(nvec, axis=1) + 1e-9)
            bonnet = (np.abs(nz) > 0.55) & (fc[:, 2] < roof_z - 0.15)
            tailgate = (nx < -0.45) | (fc[:, 0] < glass_rear + 0.02)
            hi_f = ((fc[:, 2] > belt) & (fc[:, 0] < glass_front + 0.35) & ~tailgate & ~bonnet) | \
                   ((fc[:, 2] > roof_z - 0.08) & ~tailgate)
            if hi_f.any():
                out.append((trimesh.Trimesh(g.vertices, g.faces[hi_f], process=False), "roof", False, None))
            if (~hi_f).any():
                out.append((trimesh.Trimesh(g.vertices, g.faces[~hi_f], process=False), "body", False, None))
            continue
        out.append((g, m if m in PAINT else "black", wheel, centre))
    rear_axle_x = rx - axle_mid
    roof_top = float(roof_z)
    info = {"front_sign": front_sign, "length": float(np.ptp(allc[:, 0])), "height": float(roof_top),
            "rear_axle_x": float(rear_axle_x), "wheelbase_model": float(fx - rx), "belt": float(belt),
            "n_wheel_parts": len(cands)}
    return out, info


def rig_parts(rear_axle_x, roof_top):
    """Roof rack, VLS-128 mast, 8 pano cameras, mirror lidars (CARLA vehicle axes)."""
    mast_x = rear_axle_x + MAST_FROM_REAR_AXLE
    rack_z = roof_top + 0.05
    P = []

    def box(ext, at, spec, rot=None):
        m = trimesh.creation.box(extents=ext)
        if rot is not None:
            m.apply_transform(rot)
        m.apply_translation(at)
        P.append((m, spec))

    def cyl(r, h, at, spec):
        m = trimesh.creation.cylinder(radius=r, height=h, sections=40)
        m.apply_translation(at)
        P.append((m, spec))

    # rack: a low aluminium grid over most of the roof (photo: ~8 cm above the roof)
    rack_z = roof_top + 0.08
    x0, x1 = mast_x - 0.95, mast_x + 0.55
    for y in (-0.62, -0.2, 0.2, 0.62):
        box((x1 - x0, 0.04, 0.04), ((x0 + x1) / 2, y, rack_z), RIG_GREY)
    for x in np.linspace(x0, x1, 6):
        box((0.04, 1.28, 0.04), (x, 0, rack_z + 0.005), RIG_BLACK)
    for x in (x0 + 0.05, x1 - 0.05):
        for y in (-0.6, 0.6):
            box((0.05, 0.05, 0.08), (x, y, rack_z - 0.05), RIG_BLACK)
    # GPS puck (white dome) toward the back of the rack
    puck = trimesh.creation.icosphere(subdivisions=2, radius=0.07)
    puck.apply_scale((1.0, 1.0, 0.45))
    puck.apply_translation((x0 + 0.25, -0.35, rack_z + 0.04))
    P.append((puck, RIG_WHITE))
    # the "stool": a short four-legged platform holding the camera ring, lidar right on top
    stool_z = rack_z + 0.12
    box((0.5, 0.5, 0.02), (mast_x, 0, stool_z), RIG_BLACK)
    for dx in (-0.22, 0.22):
        for dy in (-0.22, 0.22):
            box((0.03, 0.03, stool_z - rack_z), (mast_x + dx, dy, (stool_z + rack_z) / 2), RIG_BLACK)
    ring_z = stool_z + 0.05
    lidar_z = stool_z + 0.16
    cyl(0.045, lidar_z - stool_z - 0.06, (mast_x, 0, (stool_z + lidar_z - 0.06) / 2), RIG_BLACK)
    # VLS-128: dark body with a lighter cap
    cyl(0.083, 0.11, (mast_x, 0, lidar_z), RIG_GREY)
    cyl(0.085, 0.03, (mast_x, 0, lidar_z + 0.07), RIG_BLACK)
    # 8 cameras around the ring, facing out
    for k in range(8):
        a = -math.pi / 4 - k * 2 * math.pi / 8
        cx, cy = CAM_RING_R * math.cos(a), -CAM_RING_R * math.sin(a)  # ROS y-left -> CARLA y-right
        rot = trimesh.transformations.rotation_matrix(math.atan2(cy, cx), (0, 0, 1))
        box((0.11, 0.06, 0.06), (mast_x + cx, cy, ring_z), RIG_BLACK, rot)
    # mirror lidars on white cone brackets
    for dx, dy, z in MIRROR_LIDARS:
        x = mast_x + dx
        cone = trimesh.creation.cone(radius=0.06, height=0.12, sections=32)
        cone.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0)))
        cone.apply_translation((x, dy, z - 0.02))
        P.append((cone, RIG_WHITE))
        cyl(0.05, 0.07, (x, dy, z + 0.065), RIG_GREY)
    return P


# CARLA (left-handed: x fwd, y right, z up) camera coords -> OpenGL (x right, y up, z back)
_GL = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


class KiaRenderer:
    def __init__(self, width, height, K, sun_az_alt=(250.0, 12.0)):
        import pyrender
        self.pr = pyrender
        self.W, self.H = width, height
        parts, self.info = load_parts()
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.55, 0.52, 0.50])
        self.body_nodes, self.wheel_nodes = [], []
        for g, key, wheel, centre in parts:
            mesh = pyrender.Mesh.from_trimesh(g, material=_mat(PAINT[key], key), smooth=False)
            node = self.scene.add(mesh)
            (self.wheel_nodes if wheel else self.body_nodes).append((node, centre))
        # indicators: amber front corner lamps + the lower part of each tail-lamp cluster, split by side;
        # a bright copy sits 3 mm proud of the lamp and is shown only while that side is flashing
        self.blink_nodes = {"left": [], "right": []}
        lamp = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1.0, 0.55, 0.08, 1.0), emissiveFactor=(1.0, 0.55, 0.05),
                                                  metallicFactor=0.0, roughnessFactor=0.4, doubleSided=True)
        for g, key, wheel, centre in parts:
            if wheel or key not in ("orange", "material"):
                continue
            tri = g.vertices[g.faces]
            fc = tri.mean(1)
            # rear: lower section of each tail-lamp cluster only (not the bumper reflectors below)
            sel_zone = (fc[:, 2] > 0.9) & (fc[:, 2] < 1.14) & (np.abs(fc[:, 1]) > 0.45) if key == "material" \
                else np.ones(len(fc), bool)
            for side, sgn in (("left", -1), ("right", 1)):
                sel = sel_zone & (np.sign(fc[:, 1]) == sgn)
                if sel.sum() < 4:
                    continue
                sub = trimesh.Trimesh(g.vertices, g.faces[sel], process=False)
                sub.remove_unreferenced_vertices()
                # push outward from the car (normals in this model are not reliably outward)
                out_x = 0.012 if sub.vertices[:, 0].mean() > 0 else -0.012
                sub.vertices = sub.vertices + np.array([out_x, 0.006 * (1 if side == "right" else -1), 0.0])
                node = self.scene.add(pyrender.Mesh.from_trimesh(sub, material=lamp, smooth=False))
                node.mesh.is_visible = False
                self.blink_nodes[side].append(node)
        for g, spec in rig_parts(self.info["rear_axle_x"], self.info["height"]):
            node = self.scene.add(pyrender.Mesh.from_trimesh(g, material=_mat(spec), smooth=False))
            self.body_nodes.append((node, None))
        from pyrender.camera import IntrinsicsCamera
        cam = IntrinsicsCamera(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], znear=0.3, zfar=300)
        self.scene.add(cam, pose=np.eye(4))
        az, alt = map(math.radians, sun_az_alt)
        self.sun_dir_world = -np.array([math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)])
        self.sun = self.scene.add(pyrender.DirectionalLight(color=[1.0, 0.86, 0.72], intensity=3.6))
        self.fill = self.scene.add(pyrender.DirectionalLight(color=[0.75, 0.82, 1.0], intensity=1.6))   # sky
        self.r = pyrender.OffscreenRenderer(width, height)

    def _look(self, d):
        """Pose for a directional light shining along world direction d (already in GL camera space)."""
        z = -d / np.linalg.norm(d)
        x = np.cross([0, 1, 0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, :3] = np.c_[x, y, z]
        return T

    def render(self, cam_to_world, ego_to_world, wheel_angle=0.0, steer=0.0, blink=None, blink_on=False):
        """Returns RGBA (uint8) and depth (metres, 0 = empty). blink: None, "left" or "right"."""
        view = _GL @ np.linalg.inv(np.array(cam_to_world))
        M = view @ np.array(ego_to_world)
        for node, _ in self.body_nodes:
            self.scene.set_pose(node, M)
        for side, nodes in self.blink_nodes.items():
            for node in nodes:
                node.mesh.is_visible = bool(blink == side and blink_on)
                self.scene.set_pose(node, M)
        for node, c in self.wheel_nodes:
            T = trimesh.transformations.translation_matrix(c)
            spin = trimesh.transformations.rotation_matrix(-wheel_angle, (0, 1, 0))
            yaw = trimesh.transformations.rotation_matrix(steer if c[0] > 0 else 0.0, (0, 0, 1))
            self.scene.set_pose(node, M @ T @ yaw @ spin @ np.linalg.inv(T))
        R3 = view[:3, :3]
        self.scene.set_pose(self.sun, self._look(R3 @ self.sun_dir_world))
        self.scene.set_pose(self.fill, self._look(R3 @ np.array([0.0, 0.0, -1.0])))  # straight down
        color, depth = self.r.render(self.scene, flags=self.pr.RenderFlags.RGBA | self.pr.RenderFlags.SHADOWS_DIRECTIONAL)
        return color, depth

    def close(self):
        self.r.delete()


def ego_matrix(x, y, z, yaw, pitch=0.0, roll=0.0):
    """CARLA actor transform as a 4x4 (same convention as carla.Transform.get_matrix())."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0, 0, 0, 1.0]])


def contact_shadow(img, cam_to_world, ego_to_world, K, sx=1.0, sy=1.0, half_len=1.95, half_w=0.88, alpha=185):
    """Soft dark footprint under the car so it sits on the road (PIL RGBA image, drawn in place)."""
    from PIL import Image, ImageDraw, ImageFilter
    E = np.array(ego_to_world)
    corners = np.array([[half_len, -half_w, 0.03, 1], [half_len, half_w, 0.03, 1],
                        [-half_len, half_w, 0.03, 1], [-half_len, -half_w, 0.03, 1]]).T
    pw = (E @ corners).T
    pc = (np.linalg.inv(np.array(cam_to_world)) @ pw.T).T
    if (pc[:, 0] < 0.5).any():
        return
    u = K[0][0] * pc[:, 1] / pc[:, 0] + K[0][2]
    v = -K[1][1] * pc[:, 2] / pc[:, 0] + K[1][2]
    pts = [(a * sx, b * sy) for a, b in zip(u, v)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 40
    x0, y0 = int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad))
    x1, y1 = int(min(img.width, max(xs) + pad)), int(min(img.height, max(ys) + pad))
    if x1 <= x0 or y1 <= y0:
        return
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).polygon([(p[0] - x0, p[1] - y0) for p in pts], fill=alpha)
    m = m.filter(ImageFilter.GaussianBlur(9))  # deep enough to hide the small car's wheels under the Kia
    shade = Image.new("RGBA", m.size, (0, 0, 0, 0))
    shade.putalpha(m)
    img.alpha_composite(shade, (x0, y0))
