from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf, mp

def calculate_homolumo_gap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    atom_str = ""
    for atom_symbol, atom_coord in zip(atom_symbols, atom_coords):
        atom_str += atom_symbol + " " + " ".join(map(str, atom_coord)) + "; "
    atom_str = atom_str[:-2]  # Remove the last "; "
    
    # create a pyscf molecule object
    pyscf_mol = gto.M(atom=atom_str, basis='6-31G*', unit='Angstrom')
    
    mf = scf.RHF(pyscf_mol)
    mf.kernel()
    
    mo_energy = mf.mo_energy
    
    if pyscf_mol.spin == 0:
        homo_idx = pyscf_mol.nelectron // 2 - 1
        lumo_idx = pyscf_mol.nelectron // 2
    else:
        homo_idx = pyscf_mol.nelectron // 2
        lumo_idx = homo_idx + 1
    
    # now we can finally calculate the gap 
    gap = mo_energy[lumo_idx] - mo_energy[homo_idx]
    return gap

smiles = 'c1ccccc1'  # Benzene
print(calculate_homolumo_gap(smiles))

