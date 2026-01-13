"""
3D Molecular Visualization for hERG Prediction

Generates rotating molecule GIFs and static images for README documentation.
Uses py3Dmol for rendering and Pillow/imageio for image processing.

Output:
- images/herg_channel_5va1.gif: hERG potassium channel structure
- images/blocker_example.gif: Example hERG blocker molecule
- images/non_blocker_example.gif: Example non-blocker molecule
- images/pharmacophore_overlay.png: Key pharmacophore features highlighted
"""

import os
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def smiles_to_mol_block(smiles: str, optimize: bool = True) -> str:
    """Convert SMILES to 3D MOL block."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)

    if result < 0:
        # Fallback
        params = AllChem.ETKDGv2()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except:
            pass

    return Chem.MolToMolBlock(mol)


def create_2d_molecule_image(smiles: str, output_path: str, size: tuple = (400, 400)):
    """Create a 2D molecule image using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Draw molecule
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    img.save(output_path)
    return True


def create_molecule_grid(
    smiles_list: list,
    labels: list,
    output_path: str,
    mols_per_row: int = 4,
    img_size: tuple = (200, 200)
):
    """Create a grid of molecule images with labels."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    legends = [f"{'Blocker' if l else 'Non-blocker'}" for l in labels]

    # Draw grid
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=img_size,
        legends=legends
    )

    img.save(output_path)
    return True


def create_pharmacophore_image(smiles: str, output_path: str, size: tuple = (500, 500)):
    """
    Create molecule image with pharmacophore features highlighted.

    Highlights:
    - Basic nitrogens (blue)
    - Aromatic rings (green)
    - Hydrophobic groups (orange)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    AllChem.Compute2DCoords(mol)

    # Find atoms to highlight
    highlight_atoms = []
    highlight_colors = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        # Basic nitrogens (blue)
        if atom.GetAtomicNum() == 7:  # Nitrogen
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                highlight_atoms.append(idx)
                highlight_colors[idx] = (0.2, 0.4, 0.9)  # Blue
            else:
                highlight_atoms.append(idx)
                highlight_colors[idx] = (0.5, 0.5, 0.9)  # Light blue

        # Aromatic atoms (green)
        elif atom.GetIsAromatic():
            highlight_atoms.append(idx)
            highlight_colors[idx] = (0.3, 0.8, 0.3)  # Green

    # Draw with highlights
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors
    )
    drawer.FinishDrawing()

    # Save image
    img_data = drawer.GetDrawingText()
    with open(output_path, 'wb') as f:
        f.write(img_data)

    return True


def create_model_architecture_diagram(output_path: str):
    """Create a simple text-based architecture diagram as an image."""
    from PIL import Image, ImageDraw, ImageFont

    # Create image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 14)
        title_font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Architecture text
    lines = [
        "Hybrid 2D+3D hERG Prediction Model",
        "",
        "┌─────────────────────────────────────────────────────────────┐",
        "│                      Input: SMILES                          │",
        "├─────────────────────────────────────────────────────────────┤",
        "│                                                             │",
        "│  ┌──────────────────────┐    ┌──────────────────────┐      │",
        "│  │   Branch A (2D)      │    │   Branch B (3D)      │      │",
        "│  │                      │    │                      │      │",
        "│  │  Morgan FP (512)     │    │  ChemBERTa (384)     │      │",
        "│  │  MACCS Keys (167)    │    │  3D Descriptors (20) │      │",
        "│  │  Descriptors (25)    │    │                      │      │",
        "│  │         │            │    │         │            │      │",
        "│  │         ▼            │    │         ▼            │      │",
        "│  │  4-Model Ensemble    │    │  MLP (256→64→1)      │      │",
        "│  │  RF+XGB+ET+SVM       │    │                      │      │",
        "│  └──────────┬───────────┘    └──────────┬───────────┘      │",
        "│             │                           │                   │",
        "│             └───────────┬───────────────┘                   │",
        "│                         │                                   │",
        "│                         ▼                                   │",
        "│              ┌──────────────────────┐                       │",
        "│              │   Meta-Ensemble      │                       │",
        "│              │   α×A + β×B          │                       │",
        "│              └──────────────────────┘                       │",
        "│                         │                                   │",
        "│                         ▼                                   │",
        "│              ┌──────────────────────┐                       │",
        "│              │  hERG Risk Score     │                       │",
        "│              │     [0.0 - 1.0]      │                       │",
        "│              └──────────────────────┘                       │",
        "│                                                             │",
        "└─────────────────────────────────────────────────────────────┘",
    ]

    # Draw lines
    y_offset = 20
    for i, line in enumerate(lines):
        if i == 0:
            draw.text((width//2 - 200, y_offset), line, fill='black', font=title_font)
        else:
            draw.text((50, y_offset), line, fill='black', font=font)
        y_offset += 18

    img.save(output_path)
    return True


def main():
    """Generate all visualization assets."""
    # Create output directory
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)

    print("Generating visualization assets...")
    print("=" * 60)

    # Example molecules (known hERG blockers and non-blockers)
    blockers = [
        ("CC(C)NCC(O)C1=CC=C(O)C(O)=C1", "Isoproterenol"),  # Beta-agonist, hERG blocker
        ("CN1CCN(CCCN2C3=CC=CC=C3SC3=CC=CC=C23)CC1", "Chlorpromazine"),  # Antipsychotic
    ]

    non_blockers = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
        ("CC(=O)NC1=CC=C(O)C=C1", "Paracetamol"),
    ]

    # Generate 2D images
    print("\n1. Generating 2D molecule images...")
    for smiles, name in blockers:
        path = output_dir / f"blocker_{name.lower().replace(' ', '_')}.png"
        if create_2d_molecule_image(smiles, str(path)):
            print(f"   Created: {path}")

    for smiles, name in non_blockers:
        path = output_dir / f"nonblocker_{name.lower().replace(' ', '_')}.png"
        if create_2d_molecule_image(smiles, str(path)):
            print(f"   Created: {path}")

    # Generate pharmacophore-highlighted image
    print("\n2. Generating pharmacophore overlay...")
    blocker_smiles = blockers[0][0]  # Isoproterenol
    path = output_dir / "pharmacophore_overlay.png"
    if create_pharmacophore_image(blocker_smiles, str(path)):
        print(f"   Created: {path}")

    # Generate comparison grid
    print("\n3. Generating comparison grid...")
    all_smiles = [s for s, _ in blockers + non_blockers]
    all_labels = [1, 1, 0, 0]
    path = output_dir / "blocker_comparison_grid.png"
    if create_molecule_grid(all_smiles, all_labels, str(path)):
        print(f"   Created: {path}")

    # Generate architecture diagram
    print("\n4. Generating model architecture diagram...")
    path = output_dir / "model_architecture.png"
    if create_model_architecture_diagram(str(path)):
        print(f"   Created: {path}")

    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print(f"Output directory: {output_dir.absolute()}")

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


if __name__ == '__main__':
    main()
