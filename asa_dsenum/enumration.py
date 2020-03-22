from asa_dsenum import generate_HNF, generate_superlattice, permutation
import numpy as np
import spglib
from copy import deepcopy
import time
from tqdm import tqdm

from asa_dsenum.generate_HNF import generate_all_superlattices, reduce_HNF_list_by_parent_lattice_symmetry
from asa_dsenum.generate_superlattice import get_superlattice
from asa_dsenum.permutation import gene_trans, get_trans_perms, shin_get_permutation, make_candidate, unique, superperiodic_unique








def ds_enumration(base_structure, index):
    """[summary]

	Arguments:
		base_structure {[list]} -- [vectors, coords, species]
		index {[int]} -- [structure_size]
	"""
    start = time.time()

    base_structure_symmetry = spglib.get_symmetry(base_structure)
    HNF_list = generate_all_superlattices(index)
    list_reduced_HNF = reduce_HNF_list_by_parent_lattice_symmetry(HNF_list, base_structure_symmetry["rotations"])

    number = 0
    for reduced_HNF in tqdm(list_reduced_HNF):
        ds_list = list()

        parent_lattice = get_superlattice(base_structure, reduced_HNF, index)
        set_of_translations = gene_trans(reduced_HNF)
        dic_zahyou = dict()

        for i in range(len(parent_lattice[1])):
            dic_zahyou[i] = parent_lattice[1][i]

        set_of_transtikans = get_trans_perms(dic_zahyou, set_of_translations)
        goalen = shin_get_permutation(parent_lattice, base_structure_symmetry, reduced_HNF)

        for fuga in range(1, index):
            omomi4 = make_candidate(parent_lattice, fuga)
            asa = unique(goalen, omomi4)
            ru = superperiodic_unique(set_of_transtikans, asa)
            ds_list.extend(ru)

        number += len(ds_list)
    return  ds_list
