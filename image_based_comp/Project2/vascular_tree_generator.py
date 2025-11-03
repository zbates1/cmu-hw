"""
Vascular Tree Generator for Liver Vasculature
==============================================

This script generates synthetic vascular trees using a simplified hybrid approach
combining Murray's law for vessel radii and space-filling heuristics.

Based on:
- Correa-Alfonso et al. (2022) - Mesh-based model of liver vasculature
- Murray (1926) - Physiological principle of minimum work

Author: CMU Image-Based Computational Modeling Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class VesselSegment:
    """Represents a single vessel segment in the vascular tree."""
    start_point: np.ndarray  # 3D coordinates
    end_point: np.ndarray    # 3D coordinates
    radius: float            # vessel radius in mm
    generation: int          # generation level (0 = root)
    parent_id: Optional[int] = None  # ID of parent segment
    segment_id: int = -1     # unique identifier

    def length(self) -> float:
        """Calculate segment length."""
        return np.linalg.norm(self.end_point - self.start_point)

    def direction(self) -> np.ndarray:
        """Get normalized direction vector."""
        vec = self.end_point - self.start_point
        return vec / (np.linalg.norm(vec) + 1e-10)

    def volume(self) -> float:
        """Calculate vessel segment volume."""
        return np.pi * self.radius**2 * self.length()

    def resistance(self, viscosity: float = 0.0035) -> float:
        """
        Calculate hydraulic resistance using Poiseuille's law.

        R = 8*mu*L / (pi*r^4)

        Args:
            viscosity: blood viscosity in Pa*s (default: 0.0035 Pa*s)

        Returns:
            resistance in Pa*s/m^3
        """
        L = self.length() / 1000.0  # convert mm to m
        r = self.radius / 1000.0     # convert mm to m
        return 8 * viscosity * L / (np.pi * r**4)


class VascularTree:
    """Vascular tree structure with generation and analysis methods."""

    def __init__(self, root_position: np.ndarray, root_radius: float, domain_bounds: np.ndarray):
        """
        Initialize vascular tree.

        Args:
            root_position: 3D coordinates of root (inlet)
            root_radius: radius of root vessel in mm
            domain_bounds: [[xmin, xmax], [ymin, ymax], [zmin, zmax]] in mm
        """
        self.segments: List[VesselSegment] = []
        self.root_position = np.array(root_position, dtype=float)
        self.root_radius = root_radius
        self.domain_bounds = np.array(domain_bounds, dtype=float)
        self.segment_counter = 0

    def add_segment(self, segment: VesselSegment) -> int:
        """Add a segment to the tree and assign ID."""
        segment.segment_id = self.segment_counter
        self.segments.append(segment)
        self.segment_counter += 1
        return segment.segment_id

    def is_inside_domain(self, point: np.ndarray) -> bool:
        """Check if point is inside the domain."""
        for i in range(3):
            if point[i] < self.domain_bounds[i, 0] or point[i] > self.domain_bounds[i, 1]:
                return False
        return True

    def get_terminal_segments(self) -> List[VesselSegment]:
        """Get all terminal (leaf) segments."""
        child_ids = {seg.segment_id for seg in self.segments if seg.parent_id is not None}
        parent_ids = {seg.parent_id for seg in self.segments if seg.parent_id is not None}
        terminal_ids = child_ids - parent_ids
        return [seg for seg in self.segments if seg.segment_id in terminal_ids]

    def total_volume(self) -> float:
        """Calculate total volume of all vessels."""
        return sum(seg.volume() for seg in self.segments)

    def total_resistance(self) -> float:
        """Calculate total hydraulic resistance."""
        terminals = self.get_terminal_segments()
        if not terminals:
            return 0.0
        # Resistance of parallel terminals: 1/R_total = sum(1/R_i)
        return 1.0 / sum(1.0 / seg.resistance() for seg in terminals)


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate vector around axis by angle using Rodrigues' rotation formula.

    Args:
        vec: vector to rotate
        axis: rotation axis (will be normalized)
        angle: rotation angle in radians

    Returns:
        rotated vector
    """
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    return (vec * np.cos(angle) +
            np.cross(axis, vec) * np.sin(angle) +
            axis * np.dot(axis, vec) * (1 - np.cos(angle)))


def generate_bifurcation_directions(parent_direction: np.ndarray,
                                    angle: float = 35.0,
                                    randomness: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two child directions for a bifurcation.

    Args:
        parent_direction: normalized direction vector of parent
        angle: nominal branching angle in degrees
        randomness: random variation factor (0 = deterministic, 1 = highly random)

    Returns:
        (direction1, direction2) normalized vectors
    """
    angle_rad = np.deg2rad(angle)

    # Add random variation to angle
    angle1 = angle_rad + np.random.uniform(-randomness, randomness) * angle_rad
    angle2 = angle_rad + np.random.uniform(-randomness, randomness) * angle_rad

    # Create perpendicular plane
    if abs(parent_direction[2]) < 0.9:
        perp = np.cross(parent_direction, np.array([0, 0, 1]))
    else:
        perp = np.cross(parent_direction, np.array([0, 1, 0]))
    perp = perp / np.linalg.norm(perp)

    # Random rotation in perpendicular plane
    rotation_angle = np.random.uniform(0, 2*np.pi)
    perp = rotate_vector(perp, parent_direction, rotation_angle)

    # Generate two child directions
    dir1 = rotate_vector(parent_direction, perp, angle1)
    dir2 = rotate_vector(parent_direction, perp, -angle2)

    # Normalize
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)

    return dir1, dir2


def apply_murrays_law(parent_radius: float, asymmetry: float = 1.0) -> Tuple[float, float]:
    """
    Calculate child radii using Murray's law: r_parent^3 = r_child1^3 + r_child2^3

    Args:
        parent_radius: radius of parent vessel
        asymmetry: ratio r1/r2 (1.0 = symmetric, >1 = asymmetric)

    Returns:
        (radius1, radius2) for child vessels
    """
    # Murray's law: r0^3 = r1^3 + r2^3
    # With r1 = beta * r2:
    # r0^3 = beta^3 * r2^3 + r2^3 = (beta^3 + 1) * r2^3
    # r2 = r0 / (beta^3 + 1)^(1/3)

    beta = asymmetry
    r2 = parent_radius / (beta**3 + 1)**(1.0/3.0)
    r1 = beta * r2

    return r1, r2


def generate_vascular_tree_recursive(tree: VascularTree,
                                     parent_segment: Optional[VesselSegment],
                                     start_pos: np.ndarray,
                                     start_radius: float,
                                     direction: np.ndarray,
                                     generation: int,
                                     max_generation: int,
                                     length_scale: float = 1.0,
                                     min_radius: float = 0.1) -> None:
    """
    Recursively generate vascular tree using bifurcation rules.

    Args:
        tree: VascularTree object to populate
        parent_segment: parent segment (None for root)
        start_pos: starting position for this segment
        start_radius: radius at start
        direction: growth direction
        generation: current generation level
        max_generation: maximum generation depth
        length_scale: scaling factor for segment length
        min_radius: minimum vessel radius (mm)
    """
    # Stop conditions
    if generation > max_generation or start_radius < min_radius:
        return

    # Calculate segment length (decreases with generation)
    base_length = 20.0  # mm
    length = base_length * (0.7 ** generation) * length_scale

    # Calculate end position
    end_pos = start_pos + direction * length

    # Check if end is inside domain
    if not tree.is_inside_domain(end_pos):
        return

    # Create segment
    parent_id = parent_segment.segment_id if parent_segment is not None else None
    segment = VesselSegment(
        start_point=start_pos,
        end_point=end_pos,
        radius=start_radius,
        generation=generation,
        parent_id=parent_id
    )
    seg_id = tree.add_segment(segment)

    # Generate children (bifurcation)
    if generation < max_generation:
        # Murray's law for child radii
        asymmetry = np.random.uniform(0.8, 1.2)  # slight asymmetry
        r1, r2 = apply_murrays_law(start_radius, asymmetry)

        # Generate branching directions
        angle = 35.0 + np.random.uniform(-10, 10)  # 25-45 degrees
        dir1, dir2 = generate_bifurcation_directions(direction, angle, randomness=0.3)

        # Recursively generate children
        generate_vascular_tree_recursive(
            tree, segment, end_pos, r1, dir1,
            generation + 1, max_generation, length_scale, min_radius
        )
        generate_vascular_tree_recursive(
            tree, segment, end_pos, r2, dir2,
            generation + 1, max_generation, length_scale, min_radius
        )


def generate_simple_tree(root_position: np.ndarray = np.array([0, 0, 0]),
                        root_radius: float = 5.0,
                        max_generation: int = 6,
                        domain_size: float = 100.0) -> VascularTree:
    """
    Generate a simple vascular tree.

    Args:
        root_position: 3D position of root (inlet)
        root_radius: radius of root vessel in mm
        max_generation: maximum branching depth
        domain_size: size of domain in mm (cubic domain)

    Returns:
        VascularTree object
    """
    # Define domain bounds
    domain_bounds = np.array([
        [-domain_size/2, domain_size/2],
        [-domain_size/2, domain_size/2],
        [0, domain_size]
    ])

    # Create tree
    tree = VascularTree(root_position, root_radius, domain_bounds)

    # Generate tree recursively
    initial_direction = np.array([0, 0, 1])  # grow upward (z-direction)
    generate_vascular_tree_recursive(
        tree, None, root_position, root_radius, initial_direction,
        generation=0, max_generation=max_generation
    )

    return tree


def visualize_tree_3d(tree: VascularTree,
                      color_by: str = 'generation',
                      save_path: Optional[str] = None,
                      show: bool = True) -> None:
    """
    Visualize vascular tree in 3D.

    Args:
        tree: VascularTree object
        color_by: 'generation', 'radius', or 'resistance'
        save_path: path to save figure (None = don't save)
        show: whether to display figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare color mapping
    if color_by == 'generation':
        values = [seg.generation for seg in tree.segments]
        cmap = plt.cm.viridis
        label = 'Generation'
    elif color_by == 'radius':
        values = [seg.radius for seg in tree.segments]
        cmap = plt.cm.plasma
        label = 'Radius (mm)'
    elif color_by == 'resistance':
        values = [np.log10(seg.resistance()) for seg in tree.segments]
        cmap = plt.cm.coolwarm
        label = 'log10(Resistance)'
    else:
        values = [0] * len(tree.segments)
        cmap = plt.cm.gray
        label = ''

    vmin, vmax = min(values), max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot segments
    for seg, val in zip(tree.segments, values):
        xs = [seg.start_point[0], seg.end_point[0]]
        ys = [seg.start_point[1], seg.end_point[1]]
        zs = [seg.start_point[2], seg.end_point[2]]

        color = cmap(norm(val))
        linewidth = max(0.5, min(seg.radius / 2.0, 5.0))

        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=0.8)

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Vascular Tree: {len(tree.segments)} segments', fontsize=14, fontweight='bold')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(label, fontsize=12)

    # Set equal aspect ratio
    bounds = tree.domain_bounds
    max_range = np.max([bounds[i, 1] - bounds[i, 0] for i in range(3)])
    mid_x = np.mean(bounds[0])
    mid_y = np.mean(bounds[1])
    mid_z = np.mean(bounds[2])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_tree_statistics(tree: VascularTree) -> None:
    """Print statistics about the vascular tree."""
    terminals = tree.get_terminal_segments()
    radii = [seg.radius for seg in tree.segments]
    lengths = [seg.length() for seg in tree.segments]
    generations = [seg.generation for seg in tree.segments]

    print("=" * 60)
    print("VASCULAR TREE STATISTICS")
    print("=" * 60)
    print(f"Total segments:        {len(tree.segments)}")
    print(f"Terminal segments:     {len(terminals)}")
    print(f"Maximum generation:    {max(generations)}")
    print(f"Total tree volume:     {tree.total_volume():.2f} mm³")
    print(f"Total resistance:      {tree.total_resistance():.2e} Pa·s/m³")
    print()
    print(f"Radius range:          {min(radii):.3f} - {max(radii):.3f} mm")
    print(f"Length range:          {min(lengths):.3f} - {max(lengths):.3f} mm")
    print(f"Mean radius:           {np.mean(radii):.3f} mm")
    print(f"Mean length:           {np.mean(lengths):.3f} mm")
    print("=" * 60)


def save_tree_to_json(tree: VascularTree, filename: str) -> None:
    """Save tree structure to JSON file for later use."""
    data = {
        'segments': [
            {
                'id': seg.segment_id,
                'parent_id': seg.parent_id,
                'start': seg.start_point.tolist(),
                'end': seg.end_point.tolist(),
                'radius': seg.radius,
                'generation': seg.generation
            }
            for seg in tree.segments
        ],
        'root_position': tree.root_position.tolist(),
        'root_radius': tree.root_radius,
        'domain_bounds': tree.domain_bounds.tolist()
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Tree saved to {filename}")


def main():
    """Main function demonstrating tree generation."""
    print("Generating liver vascular tree...")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate tree
    tree = generate_simple_tree(
        root_position=np.array([0, 0, 0]),
        root_radius=5.0,      # 5 mm root vessel (hepatic artery)
        max_generation=7,     # 7 levels of branching
        domain_size=100.0     # 100 mm cubic domain
    )

    # Print statistics
    print_tree_statistics(tree)
    print()

    # Visualize
    print("Creating visualizations...")
    visualize_tree_3d(tree, color_by='generation', save_path='tree_generation.png', show=False)
    visualize_tree_3d(tree, color_by='radius', save_path='tree_radius.png', show=False)
    visualize_tree_3d(tree, color_by='resistance', save_path='tree_resistance.png', show=True)

    # Save to file
    save_tree_to_json(tree, 'vascular_tree.json')

    print()
    print("Done! Check the generated PNG files and JSON file.")


if __name__ == "__main__":
    main()
