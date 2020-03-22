### generating superlattice
import numpy as np
import asa_dsenum.generate_HNF
import spglib
from copy import deepcopy

def get_superlattice_vectors(o_sublattice, HNF):
    lattice = np.asarray(o_sublattice[0])
    lattice = lattice.dot(HNF)
    lattice = lattice.T
    lattice = lattice.tolist()
    return lattice



def zurasu(motono, a, i):
    original = deepcopy(motono)
    for j in range(1, a[i]):
        vec = np.zeros(3)
        vec[i] = j
        t = deepcopy(original)
        for k in range(len(original)):
            t[k] += vec
        motono = np.concatenate([motono, t])
    return motono

def tukuru(lattice, HNF):
    a = np.sum(HNF, axis=1)
    for i in range(3):
        lattice = zurasu(lattice, a, i)
    return lattice

###　superlattice の座標を生成

def making_superlattice_coordinates(lattice, HNF):

    superlattice = tukuru(lattice[1], HNF)

    g = deepcopy(superlattice)
    renew = np.linalg.inv(HNF)#逆行列 新しいsuperlatticevector で分立座標にあらわせるように　
    renew = renew.T
    for i in range(len(superlattice)):
        g[i] = g[i].dot(renew)

    #消すべき行を記録
    h = []
    for kesu in range(len(g)):
        if  0 <= g[kesu][0] < 0.99 and 0 <= g[kesu][1] < 0.99 and 0 <= g[kesu][2] < 0.99 :
            continue
        else:
            h.append(kesu)

    new_zahyou = np.delete(g, h, 0)
    return new_zahyou

def get_gensi(lattice, index):
    gensi_list = list()
    for i in range(index):
        gensi_list += lattice[2]
    return gensi_list

#spglib に突っ込めるような最後の形に仕上げる
def get_superlattice(lattice, HNF, index):
    a = [[], [], []] # a は　新しいsuperlattice
    a[0] = get_superlattice_vectors(lattice, HNF)
    a[1] = making_superlattice_coordinates(lattice, HNF)
    a[1] = np.round(a[1], 4)
    a[1] = a[1].tolist()
    a[2] = get_gensi(lattice, index)
    a = tuple(a)
    return a