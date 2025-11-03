"""
Liver Mesh Generation and Hemodynamic Analysis
===============================================

This script:
1. Generates a vascular tree (or loads from file)
2. Creates a tetrahedral mesh of liver domain with embedded vessels
3. Performs hemodynamic analysis (pressure and flow distribution)
4. Visualizes results

Based on:
- Correa-Alfonso et al. (2022) - Mesh generation with vasculature
- Poiseuille flow in vascular networks
- Murray's law principles

Author: CMU Image-Based Computational Modeling Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

# Try to import mesh generation libraries (optional)
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    warnings.warn("meshio not available. Mesh export will be limited.")


@dataclass
class Node:
    """Vascular network node."""
    id: int
    position: np.ndarray
    pressure: float = 0.0  # Pa
    is_inlet: bool = False
    is_outlet: bool = False


@dataclass
class Edge:
    """Vascular network edge (segment)."""
    id: int
    node1_id: int
    node2_id: int
    radius: float  # mm
    length: float  # mm
    flow: float = 0.0  # m^3/s


class VascularNetwork:
    """Vascular network for hemodynamic analysis."""

    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.node_map: Dict[int, Node] = {}
        self.edge_map: Dict[int, Edge] = {}

    def add_node(self, node: Node) -> None:
        """Add a node to the network."""
        self.nodes.append(node)
        self.node_map[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the network."""
        self.edges.append(edge)
        self.edge_map[edge.id] = edge

    def get_node_neighbors(self, node_id: int) -> List[Tuple[int, int]]:
        """
        Get neighboring nodes and connecting edge IDs.

        Returns:
            List of (neighbor_node_id, edge_id) tuples
        """
        neighbors = []
        for edge in self.edges:
            if edge.node1_id == node_id:
                neighbors.append((edge.node2_id, edge.id))
            elif edge.node2_id == node_id:
                neighbors.append((edge.node1_id, edge.id))
        return neighbors

    def get_inlet_nodes(self) -> List[Node]:
        """Get all inlet nodes."""
        return [n for n in self.nodes if n.is_inlet]

    def get_outlet_nodes(self) -> List[Node]:
        """Get all outlet nodes."""
        return [n for n in self.nodes if n.is_outlet]


def load_tree_from_json(filename: str) -> Dict:
    """Load vascular tree from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def build_network_from_tree(tree_data: Dict) -> VascularNetwork:
    """
    Build a vascular network from tree data.

    Args:
        tree_data: dictionary with 'segments' list

    Returns:
        VascularNetwork object
    """
    network = VascularNetwork()

    # Create nodes from segment endpoints
    node_positions = {}  # position tuple -> node_id
    node_counter = 0

    for seg_data in tree_data['segments']:
        start = tuple(seg_data['start'])
        end = tuple(seg_data['end'])

        # Add start node if not exists
        if start not in node_positions:
            node = Node(id=node_counter, position=np.array(seg_data['start']))
            network.add_node(node)
            node_positions[start] = node_counter
            node_counter += 1

        # Add end node if not exists
        if end not in node_positions:
            node = Node(id=node_counter, position=np.array(seg_data['end']))
            network.add_node(node)
            node_positions[end] = node_counter
            node_counter += 1

    # Create edges from segments
    for i, seg_data in enumerate(tree_data['segments']):
        start = tuple(seg_data['start'])
        end = tuple(seg_data['end'])

        node1_id = node_positions[start]
        node2_id = node_positions[end]

        start_arr = np.array(seg_data['start'])
        end_arr = np.array(seg_data['end'])
        length = np.linalg.norm(end_arr - start_arr)

        edge = Edge(
            id=i,
            node1_id=node1_id,
            node2_id=node2_id,
            radius=seg_data['radius'],
            length=length
        )
        network.add_edge(edge)

    # Identify inlet and outlet nodes
    # Inlet: node with no incoming edges (minimum z-coordinate)
    # Outlets: nodes with no outgoing edges (maximum z-coordinate or terminal nodes)

    # Count connections per node
    connections = {n.id: 0 for n in network.nodes}
    for edge in network.edges:
        connections[edge.node1_id] += 1
        connections[edge.node2_id] += 1

    # Find root (should have only 1 connection at the start)
    z_coords = [n.position[2] for n in network.nodes]
    min_z_node = network.nodes[np.argmin(z_coords)]
    min_z_node.is_inlet = True

    # Find terminals (leaf nodes with only 1 connection)
    for node in network.nodes:
        if connections[node.id] == 1 and node.id != min_z_node.id:
            node.is_outlet = True

    print(f"Network built: {len(network.nodes)} nodes, {len(network.edges)} edges")
    print(f"Inlets: {len(network.get_inlet_nodes())}, Outlets: {len(network.get_outlet_nodes())}")

    return network


def calculate_resistance(edge: Edge, viscosity: float = 0.0035) -> float:
    """
    Calculate hydraulic resistance of an edge using Poiseuille's law.

    R = 8*mu*L / (pi*r^4)

    Args:
        edge: Edge object
        viscosity: blood viscosity in Pa*s

    Returns:
        resistance in Pa*s/m^3
    """
    L = edge.length / 1000.0  # mm to m
    r = edge.radius / 1000.0   # mm to m
    return 8 * viscosity * L / (np.pi * r**4)


def solve_hemodynamics(network: VascularNetwork,
                       inlet_pressure: float = 13332.0,  # Pa (100 mmHg)
                       outlet_pressure: float = 1333.0,   # Pa (10 mmHg)
                       viscosity: float = 0.0035) -> VascularNetwork:
    """
    Solve for pressure and flow in vascular network.

    Uses linear system based on:
    - Flow conservation at each node (Kirchhoff's current law)
    - Poiseuille flow in each edge: Q = (P1 - P2) / R

    Args:
        network: VascularNetwork object
        inlet_pressure: pressure at inlet in Pa
        outlet_pressure: pressure at outlets in Pa
        viscosity: blood viscosity in Pa*s

    Returns:
        Updated network with pressure and flow values
    """
    N = len(network.nodes)

    # Build linear system: A * P = b
    A = lil_matrix((N, N))
    b = np.zeros(N)

    # Set up equations for each node
    for node in network.nodes:
        i = node.id

        if node.is_inlet:
            # Dirichlet BC: P_inlet = fixed
            A[i, i] = 1.0
            b[i] = inlet_pressure

        elif node.is_outlet:
            # Dirichlet BC: P_outlet = fixed
            A[i, i] = 1.0
            b[i] = outlet_pressure

        else:
            # Flow conservation: sum(Q_in) = sum(Q_out)
            # Q_ij = (P_i - P_j) / R_ij
            # sum_j (P_i - P_j)/R_ij = 0
            # P_i * sum_j(1/R_ij) - sum_j(P_j/R_ij) = 0

            neighbors = network.get_node_neighbors(i)
            for neighbor_id, edge_id in neighbors:
                edge = network.edge_map[edge_id]
                R = calculate_resistance(edge, viscosity)

                A[i, i] += 1.0 / R
                A[i, neighbor_id] -= 1.0 / R

    # Solve linear system
    A_csr = A.tocsr()
    P = spsolve(A_csr, b)

    # Update node pressures
    for node in network.nodes:
        node.pressure = P[node.id]

    # Calculate flows in edges
    for edge in network.edges:
        P1 = network.node_map[edge.node1_id].pressure
        P2 = network.node_map[edge.node2_id].pressure
        R = calculate_resistance(edge, viscosity)
        edge.flow = (P1 - P2) / R  # m^3/s

    return network


def analyze_hemodynamics(network: VascularNetwork) -> Dict:
    """
    Analyze hemodynamic properties of the network.

    Returns:
        Dictionary with analysis results
    """
    pressures = [node.pressure for node in network.nodes]
    flows = [abs(edge.flow) for edge in network.edges]

    # Convert to more readable units
    pressures_mmHg = [p / 133.322 for p in pressures]  # Pa to mmHg
    flows_ml_min = [f * 1e6 * 60 for f in flows]  # m^3/s to ml/min

    # Calculate shear stress: tau = 4*mu*Q / (pi*r^3)
    shear_stresses = []
    reynolds_numbers = []
    rho_blood = 1060  # kg/m^3

    for edge in network.edges:
        r = edge.radius / 1000.0  # mm to m
        Q = abs(edge.flow)
        mu = 0.0035  # Pa*s

        tau = 4 * mu * Q / (np.pi * r**3)  # Pa
        shear_stresses.append(tau)

        # Reynolds number: Re = rho * v * d / mu = 4 * rho * Q / (pi * d * mu)
        d = 2 * r
        Re = 4 * rho_blood * Q / (np.pi * d * mu)
        reynolds_numbers.append(Re)

    inlet_nodes = network.get_inlet_nodes()
    outlet_nodes = network.get_outlet_nodes()

    total_flow = sum(abs(edge.flow) for edge in network.edges
                    if edge.node1_id == inlet_nodes[0].id)

    results = {
        'num_nodes': len(network.nodes),
        'num_edges': len(network.edges),
        'num_outlets': len(outlet_nodes),
        'pressure_range_mmHg': (min(pressures_mmHg), max(pressures_mmHg)),
        'pressure_mean_mmHg': np.mean(pressures_mmHg),
        'flow_range_ml_min': (min(flows_ml_min), max(flows_ml_min)),
        'flow_mean_ml_min': np.mean(flows_ml_min),
        'total_flow_ml_min': total_flow * 1e6 * 60,
        'shear_stress_range_Pa': (min(shear_stresses), max(shear_stresses)),
        'shear_stress_mean_Pa': np.mean(shear_stresses),
        'reynolds_max': max(reynolds_numbers),
        'reynolds_mean': np.mean(reynolds_numbers)
    }

    return results


def print_hemodynamic_results(results: Dict) -> None:
    """Print hemodynamic analysis results."""
    print("=" * 70)
    print("HEMODYNAMIC ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Network size:              {results['num_nodes']} nodes, {results['num_edges']} edges")
    print(f"Number of outlets:         {results['num_outlets']}")
    print()
    print(f"Pressure range:            {results['pressure_range_mmHg'][0]:.1f} - "
          f"{results['pressure_range_mmHg'][1]:.1f} mmHg")
    print(f"Mean pressure:             {results['pressure_mean_mmHg']:.1f} mmHg")
    print()
    print(f"Flow rate range:           {results['flow_range_ml_min'][0]:.3f} - "
          f"{results['flow_range_ml_min'][1]:.3f} ml/min")
    print(f"Mean flow rate:            {results['flow_mean_ml_min']:.3f} ml/min")
    print(f"Total inlet flow:          {results['total_flow_ml_min']:.2f} ml/min")
    print()
    print(f"Shear stress range:        {results['shear_stress_range_Pa'][0]:.2f} - "
          f"{results['shear_stress_range_Pa'][1]:.2f} Pa")
    print(f"Mean shear stress:         {results['shear_stress_mean_Pa']:.2f} Pa")
    print()
    print(f"Maximum Reynolds number:   {results['reynolds_max']:.1f}")
    print(f"Mean Reynolds number:      {results['reynolds_mean']:.1f}")
    if results['reynolds_max'] < 2300:
        print("  -> Flow is laminar (Re < 2300)")
    else:
        print("  -> WARNING: Flow may be transitional/turbulent (Re > 2300)")
    print("=" * 70)


def visualize_network_hemodynamics(network: VascularNetwork,
                                   color_by: str = 'pressure',
                                   save_path: Optional[str] = None,
                                   show: bool = True) -> None:
    """
    Visualize network with hemodynamic properties.

    Args:
        network: VascularNetwork with solved hemodynamics
        color_by: 'pressure', 'flow', or 'shear'
        save_path: path to save figure
        show: whether to display figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare color mapping
    if color_by == 'pressure':
        values = [network.node_map[e.node1_id].pressure / 133.322
                 for e in network.edges]  # mmHg
        cmap = plt.cm.coolwarm
        label = 'Pressure (mmHg)'
    elif color_by == 'flow':
        values = [abs(e.flow) * 1e6 * 60 for e in network.edges]  # ml/min
        cmap = plt.cm.viridis
        label = 'Flow Rate (ml/min)'
    elif color_by == 'shear':
        values = []
        for e in network.edges:
            r = e.radius / 1000.0
            Q = abs(e.flow)
            mu = 0.0035
            tau = 4 * mu * Q / (np.pi * r**3)
            values.append(tau)
        cmap = plt.cm.plasma
        label = 'Shear Stress (Pa)'
    else:
        values = [0] * len(network.edges)
        cmap = plt.cm.gray
        label = ''

    if len(values) > 0:
        vmin, vmax = min(values), max(values)
        if vmax > vmin:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)

    # Plot edges
    for edge, val in zip(network.edges, values):
        n1 = network.node_map[edge.node1_id]
        n2 = network.node_map[edge.node2_id]

        xs = [n1.position[0], n2.position[0]]
        ys = [n1.position[1], n2.position[1]]
        zs = [n1.position[2], n2.position[2]]

        color = cmap(norm(val))
        linewidth = max(0.5, min(edge.radius / 2.0, 5.0))

        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=0.8)

    # Mark inlet and outlet nodes
    inlet_nodes = network.get_inlet_nodes()
    outlet_nodes = network.get_outlet_nodes()

    if inlet_nodes:
        inlet_pos = np.array([n.position for n in inlet_nodes])
        ax.scatter(inlet_pos[:, 0], inlet_pos[:, 1], inlet_pos[:, 2],
                  c='red', s=100, marker='o', label='Inlet', edgecolors='black', linewidths=2)

    if outlet_nodes:
        outlet_pos = np.array([n.position for n in outlet_nodes])
        ax.scatter(outlet_pos[:, 0], outlet_pos[:, 1], outlet_pos[:, 2],
                  c='blue', s=50, marker='s', label='Outlets', alpha=0.6)

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Hemodynamic Analysis: {color_by.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(label, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def export_network_to_vtk(network: VascularNetwork, filename: str) -> None:
    """
    Export network to VTK format for visualization in ParaView.

    Args:
        network: VascularNetwork object
        filename: output VTK filename
    """
    if not MESHIO_AVAILABLE:
        print("meshio not available. Cannot export to VTK.")
        return

    # Prepare points
    points = np.array([n.position for n in network.nodes])

    # Prepare cells (lines)
    cells = [("line", np.array([[e.node1_id, e.node2_id] for e in network.edges]))]

    # Prepare point data
    point_data = {
        "pressure_Pa": np.array([n.pressure for n in network.nodes]),
        "pressure_mmHg": np.array([n.pressure / 133.322 for n in network.nodes])
    }

    # Prepare cell data
    cell_data = {
        "flow_m3_s": [np.array([e.flow for e in network.edges])],
        "flow_ml_min": [np.array([e.flow * 1e6 * 60 for e in network.edges])],
        "radius_mm": [np.array([e.radius for e in network.edges])]
    }

    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data
    )

    # Write to file
    meshio.write(filename, mesh)
    print(f"Network exported to {filename}")


def main():
    """Main function demonstrating mesh generation and hemodynamic analysis."""
    print("Liver Vascular Mesh and Hemodynamic Analysis")
    print("=" * 70)
    print()

    # Check if tree file exists
    tree_file = "vascular_tree.json"
    try:
        tree_data = load_tree_from_json(tree_file)
        print(f"Loaded vascular tree from {tree_file}")
    except FileNotFoundError:
        print(f"ERROR: {tree_file} not found.")
        print("Please run vascular_tree_generator.py first to generate the tree.")
        return

    print()

    # Build network from tree
    print("Building vascular network...")
    network = build_network_from_tree(tree_data)
    print()

    # Solve hemodynamics
    print("Solving hemodynamics...")
    print("Boundary conditions:")
    print("  Inlet pressure:  100 mmHg (13332 Pa)")
    print("  Outlet pressure: 10 mmHg (1333 Pa)")
    print("  Blood viscosity: 0.0035 Pa*s")
    print()

    network = solve_hemodynamics(
        network,
        inlet_pressure=13332.0,   # 100 mmHg
        outlet_pressure=1333.0,    # 10 mmHg
        viscosity=0.0035
    )
    print("Hemodynamics solved!")
    print()

    # Analyze results
    results = analyze_hemodynamics(network)
    print_hemodynamic_results(results)
    print()

    # Visualize
    print("Creating visualizations...")
    visualize_network_hemodynamics(network, color_by='pressure',
                                   save_path='hemodynamics_pressure.png', show=False)
    visualize_network_hemodynamics(network, color_by='flow',
                                   save_path='hemodynamics_flow.png', show=False)
    visualize_network_hemodynamics(network, color_by='shear',
                                   save_path='hemodynamics_shear.png', show=True)

    # Export to VTK
    if MESHIO_AVAILABLE:
        print()
        print("Exporting to VTK format...")
        export_network_to_vtk(network, "liver_network.vtk")
        print("  You can open liver_network.vtk in ParaView for advanced visualization")

    print()
    print("=" * 70)
    print("Analysis complete!")
    print("Generated files:")
    print("  - hemodynamics_pressure.png")
    print("  - hemodynamics_flow.png")
    print("  - hemodynamics_shear.png")
    if MESHIO_AVAILABLE:
        print("  - liver_network.vtk")
    print("=" * 70)


if __name__ == "__main__":
    main()
