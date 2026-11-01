from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D


def validate_smiles(smiles):
    """Return an RDKit molecule, or None if the SMILES is invalid."""
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def draw_2d_svg(mol, width=400, height=300):
    """Return a 2D drawing of the molecule as SVG text."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_2d_image(mol, size=(400, 300)):
    """Return a 2D drawing of the molecule as a PIL image."""
    return Draw.MolToImage(mol, size=size)


def make_3d_molblock(mol, seed=42):
    """Return a 3D MolBlock with hydrogens, or None if embedding fails."""
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, randomSeed=seed) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol_h)


def basic_properties(mol):
    """Return a dictionary of common molecular descriptors."""
    return {
        "Molecular weight": round(Descriptors.MolWt(mol), 3),
        "Exact mass": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-bond donors": Descriptors.NumHDonors(mol),
        "H-bond acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
    }


if __name__ == "__main__":
    m = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
    if m is None:
        print("Invalid SMILES")
    else:
        for name, value in basic_properties(m).items():
            print(f"{name}: {value}")
