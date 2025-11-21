from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import numpy as np

### ---------- STEP 1: Merge Molecule + Matrix ----------
def merge_structures(matrix_file, molecule_file, translation_vector=None):
    # Load structures
    matrix_struct = Structure.from_file(matrix_file)
    mol_struct = Structure.from_file(molecule_file)

    mol_cart_coords = mol_struct.cart_coords.copy()

    # Optional translation
    if translation_vector is not None:
        mol_cart_coords += np.array(translation_vector)

    # Convert molecule coords to matrix fractional basis
    triclinic_matrix = matrix_struct.lattice.matrix
    inv_triclinic = np.linalg.inv(triclinic_matrix)
    mol_frac_coords = np.dot(mol_cart_coords, inv_triclinic)

    # Add molecule atoms to matrix
    for i, site in enumerate(mol_struct.sites):
        matrix_struct.append(site.species_string, mol_frac_coords[i], coords_are_cartesian=False)

    return matrix_struct


### ---------- STEP 2: Select Atoms for Manipulation ----------
def select_atoms(struct, indices):
    """
    Extracts Cartesian positions of selected atoms.
    indices: list of atom indices (0-based, as in pymatgen Structure)
    """
    return np.array([struct.cart_coords[i] for i in indices])


### ---------- STEP 3: Rotate Selected Atoms ----------
def rotate_atoms(positions, pivot_index, axis='x', degree=-90):
    """
    Rotate atoms around a pivot atom.
    positions: np.ndarray (N,3)
    pivot_index: int, index of pivot atom in positions
    axis: 'x', 'y', 'z'
    degree: rotation angle in degrees
    """
    theta = np.radians(degree)

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]])

    R = {'x': Rx, 'y': Ry, 'z': Rz}[axis]

    # Translate pivot to origin
    pivot = positions[pivot_index]
    translated = positions - pivot

    # Apply rotation
    rotated = np.dot(translated, R.T)

    # Translate back
    return rotated + pivot


### ---------- STEP 4: Distance Constraint Candidates ----------
def generate_distance_candidates(struct, atomA_idx, atomB_idx, target_d, n_candidates=5):
    """
    Generate n_candidates random Cartesian positions for atomA, 
    at distance target_d from atomB (atomB is fixed).
    """
    posB = struct.cart_coords[atomB_idx]

    candidates = []
    for _ in range(n_candidates):
        # Random direction using normal distribution
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)  # normalize to unit vector

        new_posA = posB + target_d * vec
        candidates.append(new_posA)

    return candidates


### ---------- STEP 5: Translate Group ----------
def translate_group(positions, old_ref, new_ref):
    """
    Translate a group of atoms so that old_ref goes to new_ref.
    """
    translation = new_ref - old_ref
    return positions + translation


### ---------- STEP 6: Export Final POSCAR ----------
def update_structure(struct, indices, new_positions, output_file="POSCAR_final"):
    for i, idx in enumerate(indices):
        struct.replace(
            idx,
            struct[idx].species_string,
            new_positions[i],
            coords_are_cartesian=True
        )
    # Manually remove selective dynamics if present
    for site in struct.sites:
        if "selective_dynamics" in site.properties:
            site.properties.pop("selective_dynamics", None)

    Poscar(struct).write_file(output_file)



### ---------- MAIN WORKFLOW ----------
if __name__ == "__main__":
    # 1. Merge + Translation
    merged = merge_structures("./23_2/CONTCAR", "./CO2_molecule/CONTCAR", translation_vector=[0, 0, 0])

    # 2. Select atoms (example: first 3 atoms from molecule part = 190, 191, 192 below)
    selected_indices = [296,297,298]   # must be adapted
    selected_positions = select_atoms(merged, selected_indices)

    # 3. Rotate around 3rd atom
    rotated_positions = rotate_atoms(selected_positions, pivot_index=2, axis='y', degree=80)

    # 4. Distance constraint between atomA and fixed atomB
    atomA_idx = 296
    atomB_idx = 295
    candidates = generate_distance_candidates(merged, atomA_idx, atomB_idx, target_d=2.5)

    print("\nGenerated candidate new positions for atomA:")
    for i, cand in enumerate(candidates):
        print(f"  Candidate {i}: {cand}")

    choice = input("\nDo you want to manually enter a position? (y/n): ").strip().lower()

    if choice == "y":
        manual_input = input("Enter Cartesian coordinates as x y z (separated by spaces): ")
        new_ref = np.array([float(x) for x in manual_input.split()])

        # Compute and print distance
        posB = merged.cart_coords[atomB_idx]
        dist = np.linalg.norm(new_ref - posB)
        print(f"Distance between atomA (manual) and atomB: {dist:.3f} Å")
    else:
        idx = int(input(f"Choose candidate index (0 to {len(candidates)-1}): "))
        new_ref = candidates[idx]

    old_ref = merged.cart_coords[atomA_idx]

    # 5. Translate whole molecule to desired distance
    translated_positions = translate_group(rotated_positions, old_ref, new_ref)

    # 6. Update structure and save
    update_structure(merged, selected_indices, translated_positions, output_file="POSCAR final")
    print("POSCAR final saved")




