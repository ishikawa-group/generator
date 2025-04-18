def get_reaction_energy(
    surface=None, method="emt", steps=10, reaction_type="N2dissociation"
):
    """
    Calculate the reaction energy of N2 dissociative adsorption on surface (N2 + 2*surf -> 2N*).
    M3GNet or CHGNet can be used.

    Args:
        surface: ASE Atoms object of surface.
        method: Method for potential energy calculation; "emt", "m3gnet", or "chgnet".
        steps: Number of optimization steps.
        reaction_type: Type of reaction. "N2dissociation" or "O2dissociation".
    Returns:
        reaction energy: Reaction energy in eV.
    """
    import warnings
    import matgl
    from ase import Atom, Atoms
    from ase.build import add_adsorbate

    # from ase.visualize import view
    from ase.optimize import FIRE, FIRE2
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    from matgl.ext.ase import PESCalculator
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    warnings.simplefilter("ignore")

    surf = surface.copy()
    bare_surf = surface.copy()

    if reaction_type == "N2dissociation":
        mol = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.1)], cell=[10, 10, 10])
        add_adsorbate(surf, Atom("N"), offset=(0.66, 0.66), height=1.4)
    elif reaction_type == "O2dissociation":
        mol = Atoms("O2", positions=[(0, 0, 0), (0, 0, 1.1)], cell=[10, 10, 10])
        add_adsorbate(surf, Atom("O"), offset=(0.66, 0.66), height=1.4)
    else:
        raise ValueError("Invalid reaction type.")

    surf.pbc = True
    mol.pbc = True
    bare_surf.pbc = True

    # --- set constraints
    c_surf = FixAtoms(indices=[atom.index for atom in surf if atom.tag > 2])
    c_bare = FixAtoms(indices=[atom.index for atom in surf if atom.tag > 2])
    bare_surf.constraints = c_bare
    surf.constraints = c_surf

    # --- set calculators
    if method == "emt":
        mol.calc = EMT()
        bare_surf.calc = EMT()
        surf.calc = EMT()
    elif method == "m3gnet":
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
    else:
        raise ValueError("Invalid method.")

    # --- optimization -- FIRE2 fail?
    opt_mol = FIRE(mol, maxstep=0.2, trajectory="mol.traj")
    opt_bare = FIRE(bare_surf, maxstep=0.2, trajectory="bare_surf.traj")
    opt_surf = FIRE(surf, maxstep=0.2, trajectory="surf.traj")

    opt_mol.run(fmax=0.1, steps=steps)
    opt_bare.run(fmax=0.1, steps=steps)
    opt_surf.run(fmax=0.1, steps=steps)

    e_mol = mol.get_potential_energy()
    e_bare_surf = bare_surf.get_potential_energy()
    e_surf = surf.get_potential_energy()

    e_reac = 2.0 * e_surf - (e_mol + 2.0 * e_bare_surf)

    if reaction_type == "N2dissociation":
        print(f"Reaction energy of N2 + 2surf -> 2N* :{e_reac} eV")
    elif reaction_type == "O2dissociation":
        print(f"Reaction energy of O2 + 2surf -> 2O* :{e_reac} eV")
    else:
        raise ValueError("Invalid reaction type.")

    return e_reac
