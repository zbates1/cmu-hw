import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for 3D projection
import math

# ============================================================
# DATA STRUCTURES (3D)
# ============================================================
class Node:
    def __init__(self, xyz, pressure=None):
        self.xyz = np.array(xyz, dtype=float)
        self.P = pressure

class Edge:
    def __init__(self, i_start, i_end, L, R, Q):
        self.i_start = i_start
        self.i_end   = i_end
        self.L = L
        self.R = R
        self.Q = Q

class Tree:
    def __init__(self):
        self.nodes = []
        self.edges = []


# ============================================================
# PHYSICS HELPERS
# ============================================================
def compute_length(a, b):
    return np.linalg.norm(a - b)

def poiseuille_radius(Q, mu, L, dP):
    if dP <= 0:
        dP = 1e-6
    return ((Q * (8*mu*L)) / (math.pi * dP)) ** 0.25

def murray_radius(R_new, R_cont, exponent=3):
    return (R_new**exponent + R_cont**exponent)**(1/exponent)

def average(points):
    return np.mean(points, axis=0)


# ============================================================
# VISUALIZATION (CLICK-THROUGH)
# ============================================================
def plot_tree_3D(tree, terminals, iteration):
    plt.clf()
    ax = plt.axes(projection='3d')

    # plot remaining terminals (in red)
    if len(terminals) > 0:
        T = np.array(terminals)
        ax.scatter(T[:,0], T[:,1], T[:,2], c='r', s=20, alpha=0.4)

    # plot edges (in blue)
    for edge in tree.edges:
        n1 = tree.nodes[edge.i_start].xyz
        n2 = tree.nodes[edge.i_end].xyz
        ax.plot([n1[0], n2[0]],
                [n1[1], n2[1]],
                [n1[2], n2[2]], 'b-', linewidth=1)

    # plot nodes (in black)
    for node in tree.nodes:
        ax.scatter(node.xyz[0], node.xyz[1], node.xyz[2], c='k', s=10)

    ax.set_title(f"Iteration {iteration}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1,1,1])
    plt.draw()

    print(f"[Iteration {iteration}] Press any key or click inside the figure...")
    plt.waitforbuttonpress()  # <— CLICK-THROUGH


# ============================================================
# MAIN 3D TREE GROWTH MODEL
# ============================================================
def build_tree_3D(N_term=30, mu=3.5e-3, P_perf=60, P_term=20, Q_perf=1.0):

    # Terminal points as tuples (safe for .remove())
    terminals = [tuple(np.random.rand(3)) for _ in range(N_term)]

    # Initialize tree
    T = Tree()

    # Root near bottom center (like the hepatic system inlets, missing the logic for outlets tho)
    root_xyz = np.array([0.5, 0.5, 0.05])
    i_root = len(T.nodes)
    T.nodes.append(Node(root_xyz, pressure=P_perf)) # given P_perf to start

    Q_term = Q_perf / N_term
    min_R = 0.003

    # Initial connection: pick nearest terminal
    t0 = min(terminals, key=lambda p: compute_length(root_xyz, np.array(p)))
    t0_arr = np.array(t0)

    L0 = compute_length(root_xyz, t0_arr)
    R0 = poiseuille_radius(Q_term, mu, L0, P_perf - P_term)

    i_leaf = len(T.nodes)
    T.nodes.append(Node(t0_arr, pressure=P_term))
    T.edges.append(Edge(i_root, i_leaf, L0, R0, Q_term))
    terminals.remove(t0)

    plot_tree_3D(T, terminals, iteration=0)

    # This was the initial connection logic, now we do the connections for the remaining nodes iteratively

    # MAIN ITERATION LOOP
    for it in range(1, N_term):

        # pick terminal closest to center of mass
        com = average([node.xyz for node in T.nodes])
        t_star = min(terminals, key=lambda p: compute_length(com, np.array(p)))
        t_star_arr = np.array(t_star)

        # candidate attach sites: midpoints of edges
        candidates = []
        for e_idx, e in enumerate(T.edges):
            n1 = T.nodes[e.i_start].xyz
            n2 = T.nodes[e.i_end].xyz
            mid = 0.5 * (n1 + n2)

            L = compute_length(mid, t_star_arr)
            P_mid = 0.5 * (T.nodes[e.i_start].P + T.nodes[e.i_end].P)
            dP = P_mid - P_term

            R_new = max(poiseuille_radius(Q_term, mu, L, dP), min_R)
            candidates.append((L, mid, e_idx, R_new, P_mid))

        # choose shortest candidate
        Lbest, mid_best, e_idx_best, R_new_best, P_mid_best = \
            min(candidates, key=lambda x: x[0])

        # split selected edge
        e_old = T.edges[e_idx_best]
        n_start = T.nodes[e_old.i_start].xyz
        n_end   = T.nodes[e_old.i_end].xyz

        # create midpoint node
        i_mid = len(T.nodes)
        T.nodes.append(Node(mid_best, pressure=P_mid_best))

        # compute new lengths
        L1 = compute_length(n_start, mid_best)
        L2 = compute_length(mid_best, n_end)

        # update old edge -> first half
        old_end = e_old.i_end
        e_old.i_end = i_mid
        e_old.L = L1

        # continuation edge -> second half
        e_cont = Edge(i_mid, old_end, L2, e_old.R, e_old.Q)
        T.edges.append(e_cont)

        # connect new terminal
        i_term = len(T.nodes)
        T.nodes.append(Node(t_star_arr, pressure=P_term))
        e_new = Edge(i_mid, i_term, Lbest, R_new_best, Q_term)
        T.edges.append(e_new)

        # update continuation flow -> this keeps the total flow constant
        e_cont.Q = e_cont.Q - Q_term

        # Murray radius update
        e_old.R = murray_radius(R_new_best, e_cont.R)

        # remove terminal so you don't connect it twice
        terminals.remove(t_star)

        # visualize this iteration
        plot_tree_3D(T, terminals, iteration=it)

    return T


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    plt.ion()
    tree = build_tree_3D(N_term=30)
    print("Simulation complete. Close the plot window to exit.")
    plt.ioff()
    plt.show()
