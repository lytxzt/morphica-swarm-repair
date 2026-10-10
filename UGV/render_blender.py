#!/usr/bin/env python3
"""Blender 3D renderer for UGV recovery trajectories.

Usage:
  blender -b -P UGV/render_blender.py -- \
    --trajectory Logs/ugv/cases/case_roughland_..._seed4.json \
    --output Logs/ugv/videos/case_roughland_..._seed4.mp4 \
    --fps 24 --stride 1

Run without Blender:
  python UGV/render_blender.py --trajectory <json> --output <mp4>
  (requires: pip install imageio imageio-ffmpeg)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

def load_trajectory(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data['metadata'], data['frames']


def load_terrain(meta, trajectory_dir):
    terrain_dir = meta.get('terrain_dir', '')
    if not terrain_dir:
        return None
    td = Path(trajectory_dir).parent / terrain_dir
    if not td.is_dir():
        td = Path(terrain_dir)
    if not td.is_dir():
        return None
    h = np.load(str(td / 'heightmap.npy'))
    # Read bounds from terrain metadata.json if available (plan §5).
    meta_path = td / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            tmeta = json.load(f)
        bounds = np.array(tmeta['bounds'])
    else:
        bounds = np.array([[0, float(meta.get('N', 50) * 10)],
                           [0, float(meta.get('N', 50) * 10)]])
    return {'heightmap': h, 'bounds': bounds,
            'resolution': meta.get('terrain_resolution', 5.0)}


def _decimate_frames(frames, stride):
    """Return every `stride`-th frame, always including the last frame.

    When the trajectory terminates at connectivity (plan §9 step 12), the
    final frame is the one where K(t)=1 was reached.  Dropping it with a
    stride that doesn't land on len-1 would produce a video that ends in a
    fragmented state — misleading for recovery visualisation.
    """
    if stride <= 1 or len(frames) <= 1:
        return list(frames)
    result = [frames[i] for i in range(0, len(frames), stride)]
    if result[-1] is not frames[-1]:
        result.append(frames[-1])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# fallback: Matplotlib renderer
# ═══════════════════════════════════════════════════════════════════════════
def render_matplotlib(meta, frames, terrain_data, output_path, fps=24):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

    N = meta['N']
    n_frames = len(frames)

    # precompute colours
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(max(
        max(f['component_id']) + 1 for f in frames))]

    # ── fixed view bounds: stable camera across frames (plan §13) ──
    if terrain_data is not None:
        b = terrain_data['bounds']
        xlim = (float(b[0, 0]) - 20, float(b[0, 1]) + 20)
        ylim = (float(b[1, 0]) - 20, float(b[1, 1]) + 20)
        z_min_t, z_max_t = terrain_data['heightmap'].min(), terrain_data['heightmap'].max()
        zlim = (float(z_min_t) - 5, float(z_max_t) + 20)
    else:
        all_pos = np.vstack([np.array(f['positions']) for f in frames])
        xlim = (float(all_pos[:, 0].min()) - 20, float(all_pos[:, 0].max()) + 20)
        ylim = (float(all_pos[:, 1].min()) - 20, float(all_pos[:, 1].max()) + 20)
        zlim = (float(all_pos[:, 2].min()) - 5, float(all_pos[:, 2].max()) + 20)

    # setup figure and terrain surface (drawn once, reused)
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    if terrain_data is not None:
        h = terrain_data['heightmap']
        ny, nx = h.shape
        xs = np.linspace(b[0, 0], b[0, 1], nx)
        ys = np.linspace(b[1, 0], b[1, 1], ny)
        X, Y = np.meshgrid(xs, ys)

    want_mp4 = output_path.endswith('.mp4')
    writer = None
    png_dir = None
    if want_mp4 and HAS_IMAGEIO:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps)
    elif want_mp4:
        # imageio unavailable but MP4 was requested — render PNGs first, then
        # combine with ffmpeg so the caller gets the requested format.
        png_dir = Path(output_path).with_suffix('')
        png_dir.mkdir(parents=True, exist_ok=True)
    else:
        png_dir = Path(output_path).with_suffix('')
        png_dir.mkdir(parents=True, exist_ok=True)

    for fi, fr in enumerate(frames):
        ax.clear()
        step = fr['step']
        pos = np.array(fr['positions'])
        comp = np.array(fr['component_id'])
        edges = fr['edges']
        remain = set(fr['remain_list'])

        # terrain
        if terrain_data is not None:
            ax.plot_surface(X, Y, h, cmap='terrain', alpha=0.5,
                            linewidth=0, antialiased=True, zorder=1)

        # destroyed agents (gray X)
        destroyed = [i for i in range(N) if i not in remain]
        if destroyed:
            ax.scatter(pos[destroyed, 0], pos[destroyed, 1],
                       pos[destroyed, 2], c='gray', marker='x',
                       s=20, alpha=0.4, zorder=3)

        # remaining agents coloured by component
        for cid in range(comp.max() + 1):
            mask = np.array([(i in remain and comp[i] == cid)
                              for i in range(N)])
            if not mask.any():
                continue
            ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2],
                       c=[colors[cid]], s=60, edgecolors='black',
                       linewidths=0.3, zorder=5, alpha=0.9)

        # edges (blue lines)
        for a, b in edges:
            ax.plot([pos[a, 0], pos[b, 0]],
                    [pos[a, 1], pos[b, 1]],
                    [pos[a, 2], pos[b, 2]],
                    'b-', lw=1.0, alpha=0.6, zorder=2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        comp_count = len(set(comp[i] for i in remain if comp[i] >= 0))
        conn = 'YES' if comp_count == 1 else f'K={comp_count}'
        ax.set_title(f'{meta["algorithm"]}  {meta["terrain"]}  '
                     f'N={N}  D={meta["D"]}  '
                     f'step={step}  connected={conn}',
                     fontsize=14, fontweight='bold')

        fig.tight_layout()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())

        if writer:
            writer.append_data(img[..., :3])
        else:
            fpath = png_dir / f'frame_{fi:04d}.png'
            plt.imsave(fpath, img)

        print(f'\r  render frame {fi+1}/{n_frames}', end='', flush=True)

    print()
    if writer:
        writer.close()
        print(f'[render] video saved: {output_path}')
    elif want_mp4 and png_dir is not None:
        # imageio unavailable — combine PNG frames with external ffmpeg.
        import subprocess
        mp4_path = str(Path(output_path))
        rc = subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(Path(png_dir) / 'frame_%04d.png'),
            '-c:v', 'libopenh264', '-b:v', '2M',
            '-pix_fmt', 'yuv420p',
            mp4_path,
        ], capture_output=True, text=True)
        if rc.returncode != 0:
            print(f'[render] ffmpeg failed: {rc.stderr.strip()}')
            print(f'[render] PNG frames kept at {png_dir}/')
            sys.exit(1)
        else:
            import shutil
            shutil.rmtree(png_dir)
            print(f'[render] video saved: {output_path}')
    else:
        print(f'[render] frames saved: {png_dir}/')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Blender renderer
# ═══════════════════════════════════════════════════════════════════════════
def _setup_blender_scene(meta, frames, terrain_data, max_edges=2000):
    """Create Blender scene: terrain, agents, edge cylinders, camera, light.

    Parameters
    ----------
    frames : list of dict — already decimated by the caller (main).
    max_edges : int — cap on edge cylinder pool size (from --max-edges).
    """
    import bpy
    import math
    from mathutils import Vector, Matrix

    N = meta['N']

    # ── clear scene ──
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Scan all rendered frames for a stable global extent (plan §13).
    # Prefer terrain bounds when available; otherwise union agent positions.
    if terrain_data is not None:
        b_t = terrain_data['bounds']
        cx = float(b_t[0, 0] + b_t[0, 1]) / 2.0
        cy = float(b_t[1, 0] + b_t[1, 1]) / 2.0
        ex = float(b_t[0, 1] - b_t[0, 0])
        ey = float(b_t[1, 1] - b_t[1, 0])
        terrain_h = terrain_data['heightmap']
        z_min = float(terrain_h.min())
        z_max = float(terrain_h.max())
    else:
        all_pos = np.vstack([np.array(f['positions']) for f in frames])
        cx, cy = float(all_pos[:, 0].mean()), float(all_pos[:, 1].mean())
        ex = float(np.ptp(all_pos[:, 0]))
        ey = float(np.ptp(all_pos[:, 1]))
        z_min = float(all_pos[:, 2].min())
        z_max = float(all_pos[:, 2].max())
    extent = max(ex, ey, 50)

    # ── terrain mesh ──
    if terrain_data is not None:
        h = terrain_data['heightmap']
        b = terrain_data['bounds']
        ny, nx = h.shape
        span_x = b[0, 1] - b[0, 0]
        span_y = b[1, 1] - b[1, 0]

        print(f'[blender] terrain {nx}×{ny}, bounds {span_x:.0f}×{span_y:.0f} m, '
              f'z [{h.min():.1f}, {h.max():.1f}]')

        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=nx - 1, y_subdivisions=ny - 1,
            size=1.0, location=(0, 0, 0))
        terrain_obj = bpy.context.active_object
        terrain_obj.name = 'Terrain'
        terrain_obj.scale = (span_x, span_y, 1.0)
        terrain_obj.location = ((b[0, 0] + b[0, 1]) / 2,
                                (b[1, 0] + b[1, 1]) / 2, 0)
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(terrain_obj.data)
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            lx = (v.co.x + 0.5) * (nx - 1)
            ly = (v.co.y + 0.5) * (ny - 1)
            ix = int(np.clip(np.floor(lx), 0, nx - 1))
            iy = int(np.clip(np.floor(ly), 0, ny - 1))
            v.co.z = h[iy, ix]
        bm.to_mesh(terrain_obj.data)
        bm.free()
        terrain_obj.data.update()
        for p in terrain_obj.data.polygons:
            p.use_smooth = True

        mat = bpy.data.materials.new(name='TerrainMat')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.35, 0.55, 0.25, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.9
        terrain_obj.data.materials.append(mat)
    else:
        print('[blender] no terrain data loaded — flat ground')

    # ── agent spheres ──
    # Visual scale: real agents are 0.6 m radius, but at ~500 m terrain extent
    # that's sub-pixel.  Scale up 5-8× for visibility.
    # We use spheres rather than detailed vehicle bodies (wheels/tracks per
    # plan §13) because at this scene scale any sub-metre geometric detail
    # would be invisible; the spheres act as colored component markers.  For
    # close-up renders of individual agents, a proper unicycle body should be
    # substituted.
    phys_radius = meta.get('agent_radius', 0.6)
    vis_scale = float(meta.get('vis_scale', 0.020))
    vis_radius = max(phys_radius, extent * vis_scale)
    body_radius = vis_radius * 1.0
    body_height = vis_radius * 0.35
    print(f'[blender] agent vis radius = {vis_radius:.1f} m (physical = {phys_radius:.1f} m, scale = {vis_scale*1000:.1f}‰ extent)')

    # ── low-poly UGV chassis model (flat cylinder "puck") ──
    def _make_ugv_body(name='UGV'):
        """Create a low-poly ground-vehicle chassis: a short wide cylinder
        that reads as a UGV body, without separate wheels (avoids hierarchy
        issues with linked duplication)."""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=body_radius, depth=body_height,
            location=(0, 0, body_height / 2))
        body = bpy.context.active_object
        body.name = name
        return body

    agent_objs = []
    component_colors = [
        (0.9, 0.2, 0.2, 1.0),  # red
        (0.2, 0.6, 0.9, 1.0),  # blue
        (0.2, 0.8, 0.3, 1.0),  # green
        (0.9, 0.7, 0.1, 1.0),  # yellow
        (0.8, 0.3, 0.8, 1.0),  # purple
        (0.9, 0.5, 0.1, 1.0),  # orange
        (0.3, 0.8, 0.8, 1.0),  # cyan
        (0.7, 0.3, 0.5, 1.0),  # pink
        (0.6, 0.6, 0.6, 1.0),  # gray
        (0.4, 0.3, 0.2, 1.0),  # brown
    ]
    agent_materials = []
    for c in component_colors:
        mat = bpy.data.materials.new(name=f'CompMat_{len(agent_materials)}')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        # Use a mix of diffuse and emission so agents read clearly regardless
        # of lighting conditions.
        bsdf = nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = c
            bsdf.inputs['Roughness'].default_value = 0.5
        emit = nodes.new('ShaderNodeEmission')
        emit.inputs['Color'].default_value = c
        emit.inputs['Strength'].default_value = 1.2
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs['Fac'].default_value = 0.45
        out = nodes.get('Material Output')
        if bsdf and out:
            links.new(bsdf.outputs['BSDF'], mix.inputs[1])
            links.new(emit.outputs['Emission'], mix.inputs[2])
            links.new(mix.outputs['Shader'], out.inputs['Surface'])
        agent_materials.append(mat)
    destroyed_mat = bpy.data.materials.new(name='DestroyedMat')
    destroyed_mat.use_nodes = True
    nodes = destroyed_mat.node_tree.nodes
    links = destroyed_mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.25, 0.25, 0.25, 1.0)
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = (0.2, 0.2, 0.2, 1.0)
    emit.inputs['Strength'].default_value = 0.3
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = 0.5
    out = nodes.get('Material Output')
    if bsdf and out:
        links.new(bsdf.outputs['BSDF'], mix.inputs[1])
        links.new(emit.outputs['Emission'], mix.inputs[2])
        links.new(mix.outputs['Shader'], out.inputs['Surface'])

    # Create UGV models (flat cylinder chassis, independent copies so each
    # agent can have its own material without affecting others).
    _make_ugv_body(name='UGV_Template')
    template_obj = bpy.context.active_object
    template_obj.hide_viewport = True
    template_obj.hide_render = True

    for i in range(N):
        # Full duplicate (not linked) — each agent owns its mesh data.
        bpy.ops.object.duplicate(linked=False)
        obj = bpy.context.active_object
        obj.name = f'Agent_{i:03d}'
        obj.hide_viewport = False
        obj.hide_render = False
        agent_objs.append(obj)

    # ── edge cylinders ──
    # Pre-allocate a pool of thin cylinders — one per edge in the densest
    # frame of the trajectory, capped to keep scene size manageable.
    # For N ≥ 100 with dense CR graphs, use --max-edges to raise the cap
    # (at the cost of slower scene setup and higher memory use).  For very
    # dense graphs (> 5000 edges) Geometry Nodes would be preferred but are
    # not implemented in this script.
    max_edges_traj = max((len(fr.get('edges', [])) for fr in frames), default=0)
    edge_pool_size = min(max_edges_traj, max_edges)
    if max_edges_traj > edge_pool_size:
        print(f'[blender] warning: trajectory contains {max_edges_traj} edges '
              f'but pool is capped at {edge_pool_size}; '
              f'{max_edges_traj - edge_pool_size} edges will be dropped. '
              f'Pass --max-edges {max_edges_traj} to render all.')
    edge_cylinders = []
    edge_radius = vis_radius * 0.45
    edge_mat = bpy.data.materials.new(name='EdgeMat')
    edge_mat.use_nodes = True
    emit = edge_mat.node_tree.nodes.new('ShaderNodeEmission')
    out = edge_mat.node_tree.nodes.get('Material Output')
    if emit and out:
        emit.inputs['Color'].default_value = (0.15, 0.4, 0.95, 1.0)
        emit.inputs['Strength'].default_value = 5.0
        edge_mat.node_tree.links.new(
            emit.outputs['Emission'], out.inputs['Surface'])

    for i in range(edge_pool_size):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=edge_radius, depth=1.0, location=(0, 0, 0),
            vertices=6, end_fill_type='NGON')
        obj = bpy.context.active_object
        obj.name = f'EdgeCyl_{i:03d}'
        obj.data.materials.append(edge_mat)
        obj.hide_viewport = True
        obj.hide_render = True
        edge_cylinders.append(obj)

    # ── camera ──
    cam_dist = extent * 0.9
    cam_z = extent * 0.7
    bpy.ops.object.camera_add(location=(cx, cy - cam_dist, z_max + cam_z))
    cam = bpy.context.active_object
    cam.name = 'Camera'
    look_target = Vector((cx, cy, (z_min + z_max) / 2))
    cam_dir = look_target - cam.location
    cam.rotation_euler = cam_dir.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    print(f'[blender] camera at ({cam.location.x:.0f},{cam.location.y:.0f},{cam.location.z:.0f}), '
          f'extent={extent:.0f}m')

    # ── lighting ──
    bpy.ops.object.light_add(type='SUN', location=(cx, cy, z_max + extent * 2))
    sun = bpy.context.active_object
    sun.data.energy = 5.0
    sun.data.angle = math.radians(3)

    # ── world ──
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.05, 0.05, 0.08, 1.0)
        bg.inputs['Strength'].default_value = 0.5

    # ── text overlay (plan §13) ──
    bpy.ops.object.text_add(location=(cx - extent * 0.45,
                                      cy - extent * 0.35,
                                      z_max + extent * 0.3))
    overlay_obj = bpy.context.active_object
    overlay_obj.name = 'Overlay'
    overlay_obj.data.body = ' '  # placeholder, filled per-frame
    overlay_obj.data.size = extent * 0.03
    overlay_obj.data.align_x = 'LEFT'
    overlay_obj.data.align_y = 'TOP'
    overlay_obj.rotation_euler = cam.rotation_euler  # face camera
    overlay_mat = bpy.data.materials.new(name='OverlayMat')
    overlay_mat.use_nodes = True
    emit = overlay_mat.node_tree.nodes.new('ShaderNodeEmission')
    out = overlay_mat.node_tree.nodes.get('Material Output')
    if emit and out:
        emit.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        emit.inputs['Strength'].default_value = 2.0
        overlay_mat.node_tree.links.new(
            emit.outputs['Emission'], out.inputs['Surface'])
    overlay_obj.data.materials.append(overlay_mat)

    # ── render settings ──
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.frame_start = 0
    scene.frame_end = len(frames) - 1

    return agent_objs, edge_cylinders, overlay_obj, agent_materials, destroyed_mat, frames


def _blender_frame_handler(scene, agent_objs, edge_cylinders, overlay_obj,
                           agent_materials, destroyed_mat, meta, frames):
    """Update agent positions/colors, edge cylinders, and text overlay for current frame."""
    from mathutils import Vector, Matrix

    frame_idx = scene.frame_current
    if frame_idx >= len(frames):
        return

    fr = frames[frame_idx]
    N = len(agent_objs)
    pos = np.array(fr['positions'])
    comp = np.array(fr['component_id'])
    edges = fr['edges']
    remain = set(fr['remain_list'])

    # update agents
    for i in range(N):
        obj = agent_objs[i]
        obj.location = (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
        if i in remain and comp[i] >= 0:
            cid = comp[i] % len(agent_materials)
            if len(obj.data.materials) == 0 or \
                    obj.data.materials[0] != agent_materials[cid]:
                obj.data.materials.clear()
                obj.data.materials.append(agent_materials[cid])
            obj.hide_viewport = False
            obj.hide_render = False
            obj.scale = (1.0, 1.0, 1.0)
        else:
            # destroyed agent — show as smaller gray sphere (plan §13)
            if len(obj.data.materials) == 0 or \
                    obj.data.materials[0] != destroyed_mat:
                obj.data.materials.clear()
                obj.data.materials.append(destroyed_mat)
            obj.hide_viewport = False
            obj.hide_render = False
            obj.scale = (0.45, 0.45, 0.45)

    # update edge cylinders
    n_cyl = len(edge_cylinders)
    for ei, (a, b) in enumerate(edges):
        if ei >= n_cyl:
            break
        cyl = edge_cylinders[ei]
        pa = Vector(pos[a])
        pb = Vector(pos[b])
        mid = (pa + pb) / 2.0
        direction = pb - pa
        length = max(direction.length, 0.001)
        z_axis = direction.normalized()
        up = Vector((0, 0, 1))
        if abs(z_axis.dot(up)) > 0.999:
            up = Vector((0, 1, 0))
        x_axis = z_axis.cross(up).normalized()
        y_axis = z_axis.cross(x_axis)
        rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
        cyl.location = mid
        cyl.rotation_mode = 'QUATERNION'
        cyl.rotation_quaternion = rot.to_quaternion()
        cyl.scale = (1.0, 1.0, length)
        cyl.hide_viewport = False
        cyl.hide_render = False

    # hide remaining unused cylinders
    for ei in range(len(edges), n_cyl):
        edge_cylinders[ei].hide_viewport = True
        edge_cylinders[ei].hide_render = True

    # update text overlay (plan §13)
    comp_count = len(set(comp[i] for i in remain if comp[i] >= 0))
    connected = 'YES' if comp_count == 1 else f'K={comp_count}'
    n_remain = len(remain)
    lines = [
        f'{meta["algorithm"]}  {meta["terrain"]}  N={N}  D={meta["D"]}',
        f'step {fr["step"]}  remain {n_remain}/{N}  {connected}',
    ]
    overlay_obj.data.body = '\n'.join(lines)


def _render_blender(meta, frames, terrain_data, output_path, fps=24, max_edges=2000):
    """Render trajectory animation using Blender's internal engine.

    Renders PNG frames via Blender, then (if MP4 requested) combines them
    with ffmpeg.  Blender 5.2 removed the FFMPEG file-format enum, so the
    external-ffmpeg path is the reliable cross-version approach.
    """
    import bpy
    import subprocess
    import tempfile
    import shutil

    agent_objs, edge_cylinders, overlay_obj, agent_materials, destroyed_mat, decimated = \
        _setup_blender_scene(meta, frames, terrain_data, max_edges)

    # store data for handler
    bpy.types.Scene._ugv_agent_objs = agent_objs
    bpy.types.Scene._ugv_edge_cylinders = edge_cylinders
    bpy.types.Scene._ugv_overlay_obj = overlay_obj
    bpy.types.Scene._ugv_materials = agent_materials
    bpy.types.Scene._ugv_destroyed_mat = destroyed_mat
    bpy.types.Scene._ugv_frames = decimated
    bpy.types.Scene._ugv_meta = meta

    # register our handler without disturbing other Blender add-ons.
    def _ugv_handler(scene):
        _blender_frame_handler(
            scene, bpy.types.Scene._ugv_agent_objs,
            bpy.types.Scene._ugv_edge_cylinders,
            bpy.types.Scene._ugv_overlay_obj,
            bpy.types.Scene._ugv_materials,
            bpy.types.Scene._ugv_destroyed_mat,
            bpy.types.Scene._ugv_meta,
            bpy.types.Scene._ugv_frames)

    bpy.app.handlers.frame_change_pre.append(_ugv_handler)

    tmpdir = None  # must be bound before try so finally can read it safely
    try:
        # output
        scene = bpy.context.scene
        scene.render.fps = fps
        scene.frame_start = 0
        scene.frame_end = len(decimated) - 1

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        want_mp4 = output_path.endswith('.mp4')

        if want_mp4:
            # Render PNG frames to a temp directory, then combine with ffmpeg.
            tmpdir = Path(tempfile.mkdtemp(prefix='ugv_render_'))
            scene.render.filepath = str(tmpdir / 'frame_')
            scene.render.image_settings.file_format = 'PNG'
        else:
            # PNG output: if the user gave a path with no extension (or an
            # existing directory), treat it as a directory and create it.
            # Otherwise treat it as a filename prefix.
            suffix = out.suffix
            if suffix == '' or out.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                scene.render.filepath = str(out / 'frame_')
            else:
                scene.render.filepath = str(out.with_suffix('')) + '_'
            scene.render.image_settings.file_format = 'PNG'

        n_frames = scene.frame_end + 1
        print(f'[blender] rendering {n_frames} frames '
              f'({scene.render.resolution_x}x{scene.render.resolution_y})')
        bpy.ops.render.render(animation=True)

        if want_mp4:
            print(f'[blender] combining {n_frames} PNGs → {output_path}')
            rc = subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(tmpdir / 'frame_%04d.png'),
                '-c:v', 'libopenh264', '-b:v', '2M',
                '-pix_fmt', 'yuv420p',
                str(output_path),
            ], capture_output=True, text=True)
            if rc.returncode != 0:
                print(f'[blender] ffmpeg error:\n{rc.stderr}')
                print(f'[blender] PNG frames kept at {tmpdir}')
                sys.exit(1)
            else:
                shutil.rmtree(tmpdir)
                tmpdir = None  # prevent finally-block cleanup of already-removed dir
                print(f'[blender] done: {output_path}')
    finally:
        # remove our handler unconditionally, even if rendering failed
        if _ugv_handler in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(_ugv_handler)
        # if ffmpeg failed and left temp PNGs, keep them for recovery;
        # otherwise tmpdir was already removed (set to None above)
        if tmpdir is not None and tmpdir.exists():
            print(f'[blender] render aborted; temp PNGs at {tmpdir}')


def _positive_int(value):
    """Argparse type: require a positive integer."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f'{value} is not a positive integer')
    return ivalue


def _nonnegative_int(value):
    """Argparse type: require a non-negative integer."""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            f'{value} is not a non-negative integer')
    return ivalue


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # detect Blender environment
    try:
        import bpy
        IN_BLENDER = True
    except ImportError:
        IN_BLENDER = False

    if IN_BLENDER:
        # Parse args after '--'
        argv = sys.argv[sys.argv.index('--') + 1:] \
            if '--' in sys.argv else []
    else:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser()
    ap.add_argument('--trajectory', required=True, help='Path to trajectory JSON')
    ap.add_argument('--output', required=True, help='Output MP4 or PNG directory')
    ap.add_argument('--fps', type=_positive_int, default=24,
                    help='Video frame rate (positive integer, default 24)')
    ap.add_argument('--stride', type=_positive_int, default=1,
                    help='Render every Nth frame (default 1)')
    ap.add_argument('--engine', default='auto',
                    choices=['auto', 'blender', 'matplotlib'])
    ap.add_argument('--max-edges', type=_nonnegative_int, default=2000,
                    help='Max edge cylinders to allocate (default 2000)')
    ap.add_argument('--vis-scale', type=float, default=0.020,
                    help='Agent visual size as fraction of terrain extent (default 0.020)')
    args = ap.parse_args(argv)

    meta, frames = load_trajectory(args.trajectory)
    terrain_data = load_terrain(meta, args.trajectory)
    # Decimate once so both renderers always see the same frame set.
    frames = _decimate_frames(frames, args.stride)
    # Inject vis-scale into meta so renderers can access it.
    meta['vis_scale'] = args.vis_scale

    if args.engine == 'matplotlib':
        render_matplotlib(meta, frames, terrain_data, args.output, args.fps)
    elif args.engine == 'blender':
        if not IN_BLENDER:
            print('ERROR: --engine blender requires running inside Blender.',
                  file=sys.stderr)
            print('Run: blender -b -P UGV/render_blender.py -- '
                  '--trajectory <json> --output <mp4>',
                  file=sys.stderr)
            sys.exit(1)
        _render_blender(meta, frames, terrain_data, args.output,
                        args.fps, args.max_edges)
    else:  # auto
        if not IN_BLENDER:
            render_matplotlib(meta, frames, terrain_data, args.output, args.fps)
        else:
            _render_blender(meta, frames, terrain_data, args.output,
                            args.fps, args.max_edges)


if __name__ == '__main__':
    main()
