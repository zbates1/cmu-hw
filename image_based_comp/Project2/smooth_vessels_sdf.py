import numpy as np
import pyvista as pv
from tqdm import tqdm
from importlib import import_module

# ===============================================
# to run, do: python smooth_vessels_sdf.py
# ===============================================

# ========= PARAMETERS =========
GRID_RES = 64     # resolution of voxel grid -> doesn't seem to help than to increase computation from sdf calculations
BLEND_K = 100     # softness of smooth union -> found higher works to preserve geometry better
ISO_LEVEL = 0.0    # SDF iso-surface -> assumed this should be zero based on class

# ===============================================
# Compute SDF of a cylinder between p0 → p1
# ===============================================
def sdf_cylinder(x, p0, p1, r):
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-6:
        return 1e9

    d = d / L
    # projection of x onto cylinder axis -> linear algebra technique using a dot product with a unit vector to get projection
    t = np.dot(x - p0, d)
    t = np.clip(t, 0, L) # making sure t is between 0 and L
    proj = p0 + t * d

    radial = np.linalg.norm(x - proj)
    return radial - r


# ===============================================
# Smooth-union via log-sum-exp (soft minimum)
# ===============================================
def smooth_union_sdf(d_list, k=BLEND_K):
    # Takes in a list of SDF values and returns a smooth union
    # log-sum-exp approximation of minimum -> found this method online and tweaked the formula a bit, and still worked!
    return -np.log(np.sum(np.exp(-k * np.array(d_list)))) / k


# ===============================================
# Build SDF volume AND pressure volume, the tree comes from 'sample_3D.py'
# ===============================================
def build_sdf_volume(tree, grid_res=GRID_RES):
    # Takes in a tree of nodes and edges and builds an SDF volume
    # Extract vessel segments
    nodes = np.array([n.xyz for n in tree.nodes])
    pressures = np.array([n.P for n in tree.nodes])   # <-- pressures from tree
    edges = [(e.i_start, e.i_end, e.R) for e in tree.edges]

    # bounding box of all vessel nodes
    mn = nodes.min(axis=0) - 0.05
    mx = nodes.max(axis=0) + 0.05

    xs = np.linspace(mn[0], mx[0], grid_res)
    ys = np.linspace(mn[1], mx[1], grid_res)
    zs = np.linspace(mn[2], mx[2], grid_res)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    sdf = np.zeros_like(X)
    pressure_field = np.zeros_like(X) # to create pressure field for coloring

    print("[*] Computing SDF + pressure field (this may take 10–30 sec)...")

    for i in tqdm(range(grid_res)):
        for j in range(grid_res):
            for k in range(grid_res):

                p = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])

                d_list = []
                pressure_list = []

                # Compute cylinder SDF + pressure interpolation
                for (a, b, r) in edges:
                    d = sdf_cylinder(p, nodes[a], nodes[b], r)
                    d_list.append(d)

                    # compute normalized scalar projection t in [0,1]
                    v = nodes[b] - nodes[a]
                    L2 = np.dot(v, v)
                    if L2 < 1e-12:
                        t = 0.0
                    else:
                        t = np.dot(p - nodes[a], v) / L2
                        t = np.clip(t, 0.0, 1.0)

                    # linear pressure interpolation along the vessel axis
                    pa = pressures[a]
                    pb = pressures[b]
                    interpP = pa*(1-t) + pb*t
                    pressure_list.append(interpP)

                # smooth union for SDF
                sdf[i,j,k] = smooth_union_sdf(d_list)

                # pressure from nearest cylinder (smallest distance)
                nearest_idx = np.argmin(d_list)
                pressure_field[i,j,k] = pressure_list[nearest_idx]

    return sdf, pressure_field, xs, ys, zs


import pyvista as pv

# convert to ply, because of the easy handling of normals, no volume data like vtk though
def convert_vtu_to_ply(vtu_path, ply_path, scalar_name=None):
    # Convert VTU to PLY, this is because the PLY file makes it super easy to extract surface normal from the surface mesh
    # Load VTU
    volume = pv.read(vtu_path)

    # Extract the *outer* triangular surface
    surface = volume.extract_surface()

    # Compute normals
    surface = surface.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True
    )

    # Transfer optional scalar field
    if scalar_name is not None and scalar_name in volume.point_data:
        surface.point_data[scalar_name] = volume.point_data[scalar_name]

    # Save pressures as quality so MeshViewer can color by pressure
    surface.point_data["quality"] = surface.point_data["pressure"]


    # Save as PLY (supports normals + scalars)
    surface.save(ply_path)

    return surface


# ===============================================
# Main
# ===============================================
if __name__ == "__main__":
    print("[*] Importing 3D_sample...")
    tree = import_module("3D_sample").build_tree_3D(N_term=10)

    print("[*] Building SDF + pressure volume...")
    sdf, pressure_field, xs, ys, zs = build_sdf_volume(tree)

    print("[*] Extracting mesh via marching cubes...")
    grid = pv.RectilinearGrid(xs, ys, zs)

    # attach both fields to the grid
    grid["sdf"] = sdf.ravel(order="F")
    grid["pressure"] = pressure_field.ravel(order="F")

    # marching cubes on the SDF field
    surface = grid.contour([ISO_LEVEL], scalars="sdf")

    # smoothing surface mesh
    surface = surface.smooth(n_iter=20)

    # the surface already carries 'pressure' point_data !
    print("Surface point-data keys:", surface.point_data.keys())

    print("[*] Saving STL...")
    surface.save("smooth_vessels.stl")

    print("[*] Saving VTK with pressure for visualization...")
    surface.save("smooth_vessels_with_pressure.vtp")

    print("[DONE] smooth_vessels.stl and smooth_vessels_with_pressure.vtp created")

    print("[*] Visualizing...")
    plotter = pv.Plotter()
    plotter.add_mesh(surface, scalars="pressure", show_scalar_bar=True)
    plotter.show()

    surface = convert_vtu_to_ply(
    "smooth_vessels_with_pressure.vtp",
    "liver_surface.ply",
    scalar_name="pressure"
)


    