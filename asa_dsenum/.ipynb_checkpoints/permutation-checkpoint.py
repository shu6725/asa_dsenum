
#generate permutation
import spglib
import numpy as np
import pymatgen
from copy import deepcopy
from asa_dsenum.generate_superlattice import get_o_superlattice, get_superlattice


def is_unimodular(M: np.ndarray) -> bool:

    if np.abs(np.around(np.linalg.det(M))) == 1:

        return True

    else:

        return False
def cast_integer_matrix(arr: np.ndarray) -> np.ndarray:

    arr_int = np.around(arr).astype(np.int)

    return arr_int
def is_integer_matrix(matrix):
    for i in range(0, 3):
        for j in range(0, 3):
            if not matrix[i][j].is_integer():
                return  False
    return True




# 座標一致判定関数の試作
def judge_pos(pos1, pos2, ips=3):
    
    determinant = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2) **(1/2)
    
    if determinant < 10 ** (-ips):
        return True
    else:
        return False
    


def get_superlattice_symmetry(parent_sym, HNF):
    H = HNF
    H_inv = np.linalg.inv(H)
    Subspacegroup = dict()
    Subspacegroup["rotations"] = list()
    Subspacegroup["translations"] = list()
    for i in range(len(parent_sym["rotations"])):
        check = H_inv.dot(parent_sym["rotations"][i].dot(H))
        if is_integer_matrix(check):
            t = parent_sym["translations"][i]
            t_neo = np.dot(H_inv, t)
            Subspacegroup["rotations"].append(check)
            Subspacegroup["translations"].append(t_neo)
    return Subspacegroup

def gene_per(rotation, translation, dic_zahyou):
    """
    対象操作について置換の移り先を示す辞書を作成

    parameters
    
    dic_zahyou:元の座標セット

    retruns

    """
    
    dic2 = deepcopy(dic_zahyou)
    for l in range(len(dic_zahyou)):
        dic2[l] = (rotation.dot(dic_zahyou[l]) + translation)
        dic2[l] = np.round(dic2[l], 4)%1
        
    tin = dict()
    for l in range(len(dic_zahyou)):
        for j in range(len(dic_zahyou)):
            if judge_pos(dic2[l], dic_zahyou[j]):
                tin[l] = j
    return tin

def shin_get_permutation(o_super_sublattice,parent_sym, HNF):
    dic_zahyou = dict()                        #分率座標の行列表示
    for i in range(len(o_super_sublattice[1])):
        dic_zahyou[i] = np.asarray(o_super_sublattice[1][i])
    trans = gene_trans(HNF)
    trans_tikans = get_trans_perms(dic_zahyou, trans)
    super_point_group = get_superlattice_symmetry(parent_sym, HNF)
    
    def generate_super_perm(dic_zahyou, super_point_group):##各座標をnumpyに変換して、gene_perを使う
        jun = dict()
        for i in range(len(super_point_group["translations"])):
            jun[i] = gene_per(super_point_group["rotations"][i], super_point_group["translations"][i], dic_zahyou)
        return jun
    
    rot_permutations = generate_super_perm(dic_zahyou, super_point_group)
    
    goal_perms = generate_abs_permuatation(rot_permutations, trans_tikans)
    
    return goal_perms



def gene_trans(HNF):
    """
    HNFから並進操作の一覧を作成

    parameters:HNF

    retruns

    """
    vectors = np.zeros(3)
    H_inv = np.linalg.inv(HNF)
    hantei = True
    for dimention in range(3):
        cp_vecs = deepcopy(vectors)
        if HNF[dimention][dimention] > 1:
            for king in range(1, HNF[dimention][dimention]):
                cp_vecs2 = deepcopy(cp_vecs)
                if hantei:
                    cp_vecs2[dimention] += king
                else:
                    for num_vecs in range(cp_vecs2.shape[0]):
                        cp_vecs2[num_vecs][dimention] += king
                vectors = np.vstack((vectors, cp_vecs2))
            hantei = False
    for hoge in range(len(vectors)):
        vectors[hoge] = np.dot(H_inv, vectors[hoge])
    return vectors

def get_trans_permuation(dic_zahyou, translation):
    """
    並進操作について置換の移り先を示す辞書を作成

    parameters

    retruns

    """
    tin = dict()
    dic2 = deepcopy(dic_zahyou)
    for l in range(len(dic_zahyou)):
        dic2[l] = (dic_zahyou[l] + translation)
        dic2[l] = np.round(dic2[l], 3)%1
    for l in range(len(dic_zahyou)):
        for j in range(len(dic_zahyou)):
            if judge_pos(dic2[l], dic_zahyou[j]):
                tin[l] = j
    return tin

def get_trans_perms(dic_zahyou, translations):
    """
    並進操作について置換の移り先を示す辞書の辞書を作成

    parameters

    retruns

    """
    trans_perms = dict()
    for i in range(len(translations)):
        trans_perms[i] = get_trans_permuation(dic_zahyou, translations[i])
    return trans_perms

def kumiawase(rot_tikan, trans_tikan):
    """
    回転操作と並進操作の置換を組合す

    parameters

    retruns

    """
    hosii = dict()
    for hayashi in range(len(rot_tikan)):
        hosii[hayashi] = trans_tikan[rot_tikan[hayashi]]
    return hosii

def superperiodic_unique(trans_tikans, omomi4):
    omomi4_neo = deepcopy(omomi4)
    lis = set()
    
    for candidate in range(len(omomi4)):
        if omomi4[candidate] in omomi4_neo: #ある元の構造が生き残ってるかチェック
            id_superperiodic = False
            
            for permutation in range(1, len(trans_tikans)):#恒等操作以外の並進操作について回す
                d = dict()
                sta = ""
                for k in range(len(trans_tikans[0])):
                    d[trans_tikans[permutation][k]] = omomi4[candidate][k]
                for r in range(len(trans_tikans[0])):
                    sta += d[r]    # 置換によって生成した配列
                if int(sta) == int(omomi4[candidate]):# 作った配列と元の配列が一致、すなわちsuper_periodicやったら
                    id_superperiodic = True
                else:#操作によって元とは別の配列になったかどうか
                    if sta in omomi4_neo:#まだ省いていない構造やったら
                        omomi4_neo.remove(sta)
                        
            if not id_superperiodic:
                lis.add(omomi4[candidate])
                    
    lis = list(lis)
    return lis

def generate_abs_permuatation(parent_lattice_jun, trans_perms):
    """
    回転操作と並進操作の置換を組合せた究極の辞書を作成

    parameters

    retruns

    """
    tikan_list = list()
    for i in parent_lattice_jun:
        for j in trans_perms:
            tikan = kumiawase(parent_lattice_jun[i], trans_perms[j])
            tikan_list.append(tikan)
    return tikan_list



def generate_per(superlattice, o_sublattice):##各座標をnumpyに変換して、gene_perを使う
    parent_sym = spglib.get_symmetry(superlattice)
    tin = dict()
    arr = np.asarray(o_sublattice[1])   #lattcie no 行列表示
    dic = dict()                        #分率座標の行列表示
    for i in range(len(o_sublattice[1])):
        dic[i] = np.asarray(o_sublattice[1][i])

    jun = dict()
    for i in range(len(parent_sym["translations"])):
        jun[i] = gene_per(parent_sym["rotations"][i], parent_sym["translations"][i], dic)
    return jun

# 置換の積の式を作る [{0, 1, 2, 3}, {4, 5, 6, 7}]
def junkai(tin):
    c = [0, 1, 2, 3, 4, 5, 6, 7]
    d = []
    f = tin[0]
    k = -1
    for i in range(8):
        if i in c:
            d.append([])
            k += 1
            f = tin[i]
            d[k].append(i)
            while f in c:
                d[k].append(f)
                c.remove(f)
                f = tin[f]
    for i in range(len(d)):
        d[i] = set(d[i])
    return d
#置換の積の式の辞書を作成
def tikan():
    jun = dict()
    for i in range(len(o_subsim["translations"])):
        tin = gene_per(i)
        jun[i] = junkai(tin)
    return jun

#置換の積の式からpolyaの数え上げの定理を使って配色のパターン数を出す
def polya(jun):
    sum = 0
    for i in range(len(jun)):
        sum += (number_of_colors)**(len(jun[i]))
    polya = sum / len(jun)
    return polya

def make_candidate(o_sublattice, l):
    omomi4 = []
    for i in range(2**len(o_sublattice[2])):
        k = format(i,'b').zfill(len(o_sublattice[2]))
        sum = 0
        for m in range(len(k)):
            sum += int(k[m])
        if sum ==l :
            omomi4.append(k)
    return omomi4

def unique(parent_sym_jun, omomi4):
    """
    得られた究極の対称操作の辞書に基づいて、配列をユニークしていく

    parameters

    parent_sym_jun

    omomi4 は各空孔数ごとのcandidate集合

    retruns

    """
    omomi4_neo = deepcopy(omomi4)#copy wo sakusei
    lis = set()

    for i in range(len(omomi4)):#through all candidate
        if omomi4[i] in omomi4_neo:#kouho ga mada 生き残ってるかチェック
            
            for j in range(1, len(parent_sym_jun)):#置換操作について回す　
                d = dict()
                sta = ""
                for k in range(len(parent_sym_jun[0])):
                    d[parent_sym_jun[j][k]] = omomi4[i][k]#str辞書の作成
                for r  in range(len(parent_sym_jun[0])):#staの作成 sta is made from tikan[j]
                    sta += d[r]
                if int(sta) is not int(omomi4[i]):
                    if sta in omomi4_neo:
                        omomi4_neo.remove(sta)
                        lis.add(omomi4[i])
    lis = list(lis)
    return lis

def sucell_unique(rutile, HNF, index, parent_sym):
    parent_lattice =  get_superlattice(rutile, HNF, index)
    o_sublattice = get_o_superlattice(rutile, HNF, index)
    goalen = shin_get_permutation(o_sublattice, parent_sym, HNF)
    ds_list = list()
    for oxy_number in range(index*4):
        can = make_candidate(o_sublattice, oxy_number)
        auncko = unique(goalen, can)
        ds_list.extend(auncko)
    return ds_list









