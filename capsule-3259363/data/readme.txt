## Details of qm9s.pt file
The properties of the 130k molecules are collected in a single file. This file is in the version of torch-geometric and can be read by torch.load().
Actually, we directly use the qm9s.pt file for our training. 

## Details of qm9s_csv.zip files
To clearly present the QM9S database, the properties of every single molecule are collected in 130k separate csv files. The detailed information are explained as following:
line0:   
    colomn0: The number of the molecule in the QM9S dataset
    colomn1: The number of the molecule in the original QM9 dataset (http://quantum-machine.org/datasets)
    colomn2: The SMILES of the molecule
    colomn3: The total number of atoms in the molecule
line1: The total energy of the molecule (Unit: eV)
line2: The atomic chargeS based on natural population analyses (Unit: eletron)
line3: The dipole moment vector (Component: x, y, z) (Unit: Debye) 
line4: The quadrupole moment tensor (Component: xx, xy, yy, zx, zy, zz) (Unit: Debye*Angstrom)
line5: The octupole moment tensor (Component: xxx, xxy, yxy, yyy, xxz, yxz, yyz, zxz, zyz, zzz) (Unit: Debye*Angstrom**2)
line6: The polarizability tensor (Component: xx, xy, yy, zx, zy, zz) (Unit: Bohr**3)
line7: The first hyperpolarizability tensor (Component: xxx, xxy, yxy, yyy, xxz, yxz, yyz, zxz, zyz, zzz) (Unit: a.u.)
line8: The excitation energies for the first 10 excitation states (Unit: eV) 
line9: The transition dipole moments for the first 10 excitation states (Unit: Debye)
line10-line(10+n): The Cartesian coordinates of the molecule (Unit: Angstrom)
line(10+n)-line(10+2n): The first derivative of dipole moment with respect to atomic positions (Unit: Debye/Angstrom)
line(10+2n)-line(10+3n): The first derivative of polarizability with respect to atomic positions (Unit: Bohr**3/Angstrom)
line(10+3n)-line(10+6n): The Hessian matrix (Unit: eV/Angstrom**2)

## Details of ext_val.zip and ext_val_env.zip fille:
To validate the good transferability of DetaNet, we randomly selected 5500 molecules from PubChem (https://pubchem.ncbi.nlm.nih.gov) out of QM9S database and perform the geometrical optimization and vibrational analysis. The properties of the 5500 molecules in gas phase are collected in the ext_val.zip. 

One of the five kinds of external environments including electric fields of 0.01 or 0.02 a.u., and solvent effect with water, ethanol or dimethyl sulfoxide is randomly added on the each molecule. The molecular properties with external environmental effects is collected in ext_val_env.zip.

Both ext_val.zip and ext_val_env.zip contains 5500 separate files. The detailed information are explained as following:
line0:
    colomn0: The number of the molecule in the data
    colomn1: The SMILES of the molecule
    colomn2: The total number of atoms in the molecule
    colomn3: The external environment that affects the structure of molecules
line1-line(1+n): The Cartesian coordinates of the molecule (Unit: Angstrom)
line(1+n)-line(1+2n): The first derivative of dipole moment with respect to atomic positions (Unit: Debye/Angstrom)
line(1+2n)-line(1+3n): The first derivative of polarizability with respect to atomic positions (Unit: Bohr**3/Angstrom)
line(1+3n)-line(1+6n): The Hessian matrix (Unit: eV/Angstrom**2)
    


