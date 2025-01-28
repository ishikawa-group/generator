def get_adsorption_energy(surface=None):
    """
    Calculate the adsorption energy of N2 on Ru(0001) surface using M3GNet or CHGNet.

    Args:
        surface: Atoms
    Return:
        float: adsorption energy
    """
    import warnings
    import matgl
    from ase import Atom, Atoms
    from ase.build import add_adsorbate
    from ase.visualize import view
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms
    from matgl.ext.ase import PESCalculator
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet
    from pymatgen.io.ase import AseAtomsAdaptor

    warnings.simplefilter("ignore")

    method = "chgnet"

    surf = surface.copy()
    bare_surf = surface.copy()

    mol = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.1)], cell=[10, 10, 10])

    add_adsorbate(surf, Atom("N"), offset=(0.66, 0.66), height=1.4)

    surf.pbc = True
    mol.pbc = True
    bare_surf.pbc = True

    # --- set constraints
    c_surf = FixAtoms(indices=[atom.index for atom in surf if atom.tag > 2])
    c_bare = FixAtoms(indices=[atom.index for atom in surf if atom.tag > 2])
    bare_surf.constraints = c_bare
    surf.constraints = c_surf

    if method == "m3gnet":
        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        mol.calc = PESCalculator(potential=potential)
        bare_surf.calc = PESCalculator(potential=potential)
        surf.calc = PESCalculator(potential=potential)
    elif method == "chgnet":
        chgnet = CHGNet.load()
        potential = CHGNetCalculator(potential=chgnet, properties="energy")
        mol.calc = potential
        bare_surf.calc = potential
        surf.calc = potential

    # --- optimization
    opt_mol = BFGS(mol, trajectory="mol.traj")
    opt_mol.run(fmax=0.1, steps=100)
    opt_bare = BFGS(bare_surf, trajectory="bare_surf.traj")
    opt_bare.run(fmax=0.1, steps=100)
    opt_surf = BFGS(surf, trajectory="surf.traj")
    opt_surf.run(fmax=0.1, steps=100)

    e_mol = mol.get_potential_energy()
    e_bare_surf = bare_surf.get_potential_energy()
    e_surf = surf.get_potential_energy()

    e_ads = e_surf - (0.5*e_mol + e_bare_surf)

    print(f"Adsorption energy = {e_ads} eV")

    return e_ads
