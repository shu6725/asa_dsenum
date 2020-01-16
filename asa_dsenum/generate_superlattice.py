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
def making_superlattice(HNF):
    Sn_sublattice = [(0, 0, 0), (0.5, 0.5, 0.5)]
    O_sublattice =  [(0.3, 0.3, 0.0),(0.7, 0.7, 0.0),(0.2, 0.8, 0.5),(0.8, 0.2, 0.5)]
    Sn_superlattice = tukuru(Sn_sublattice, HNF)
    O_superlattice = tukuru(O_sublattice, HNF)
    superlattice = np.concatenate([Sn_superlattice, O_superlattice])

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

def making_o_subsuperlattice(HNF):
    O_sublattice =  [(0.3, 0.3, 0.0),(0.7, 0.7, 0.0),(0.2, 0.8, 0.5),(0.8, 0.2, 0.5)]
    O_superlattice = tukuru(O_sublattice, HNF)
    superlattice = O_superlattice

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

def get_parent_gensi(index):
    empty = []
    for i in range(index*2):
        empty.append(50)
    for j in range(index*4):
        empty.append(8)
    return empty

def get_o_sub_gensi(index):
    empty = []
    for j in range(index*4):
        empty.append(8)
    return empty

#spglib に突っ込めるような最後の形に仕上げる
def get_superlattice(parent_lattice, HNF, index):
    a = [[], [], []] # a は　新しいsuperlattice
    a[0] = get_superlattice_vectors(parent_lattice, HNF)
    a[1] = making_superlattice(HNF)
    a[1] = np.round(a[1], 4)
    a[1] = a[1].tolist()
    a[2] = get_parent_gensi(index)
    a = tuple(a)
    return a

def get_o_superlattice(parent_lattice, HNF, index):
    a = [[], [], []] # a は　新しいsuperlattice
    a[0] = get_superlattice_vectors(parent_lattice, HNF)
    a[1] = making_o_subsuperlattice(HNF)
    a[1] = np.round(a[1], 4)
    a[1] = a[1].tolist()
    a[2] = get_o_sub_gensi(index)
    a = tuple(a)
    return a
