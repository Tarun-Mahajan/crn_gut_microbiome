# import hdbscan
# from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
# import matplotlib as mpl
# import math
import numpy as np
import os
import pandas as pd
import pickle
# import random
from scipy.optimize import nnls
# from scipy.optimize import curve_fit
# from scipy.optimize import minimize
import scipy
# import scipy.cluster.hierarchy as sch
# from scipy.stats import mannwhitneyu, normaltest
import seaborn as sns
# from sklearn.cluster import KMeans
# from sklearn.cluster import affinity_propagation
# from sklearn.metrics import pairwise_distances
# from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
# from sklearn.datasets import make_regression
import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from statannotations.Annotator import Annotator
# import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
import math

def iterate_growth_ratio(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, p_, \
                         df_speciesAbun_ratio=None, power_=1.0):
    if df_speciesAbun_ratio is None:
        df_speciesAbun_split = \
            geometric_avg(df_speciesAbun_prev, df_speciesAbun_next, p=p_)
    else:
        # df_speciesAbun_split = df_speciesAbun_ratio.copy()**(1 - p_) * \
        #     df_speciesAbun_prev.copy()
        df_speciesAbun_split = \
            geometric_avg(df_speciesAbun_prev, df_speciesAbun_next, p=p_)
    df_cons_abun_prod_split = compute_cons_abun_prod(df_speciesMetab=df_speciesMetab, \
                                                     df_speciesAbun=df_speciesAbun_split, \
                                                     power_=power_)
    sample_names_split, mat_cons_abun_split_list = \
        compute_species_metab_matrix_for_nnls(df_cons_abun_prod=df_cons_abun_prod_split, 
                                              df_speciesMetab=df_speciesMetab, 
                                              num_passages=6, num_bioRep=3, 
                                              sample_names=list(df_speciesAbun_split.columns.values))
    return sample_names_split, mat_cons_abun_split_list

def iterate_growth_ratio_2_0(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, p_, \
                         df_speciesAbun_ratio=None, power_=1.0, D_=15000):
    # if df_speciesAbun_ratio is None:
    #     df_speciesAbun_split = \
    #         geometric_avg(df_speciesAbun_prev, df_speciesAbun_next, p=p_)
    # else:
    #     # df_speciesAbun_split = df_speciesAbun_ratio.copy()**(1 - p_) * \
    #     #     df_speciesAbun_prev.copy()
    #     df_speciesAbun_split = \
    #         geometric_avg(df_speciesAbun_prev, df_speciesAbun_next, p=p_)
    df_speciesAbun_split = df_speciesAbun_next / \
        (np.log(df_speciesAbun_next) + np.log(D_) - np.log(df_speciesAbun_prev))
    df_cons_abun_prod_split = compute_cons_abun_prod(df_speciesMetab=df_speciesMetab, \
                                                     df_speciesAbun=df_speciesAbun_split, \
                                                     power_=power_)
    sample_names_split, mat_cons_abun_split_list = \
        compute_species_metab_matrix_for_nnls(df_cons_abun_prod=df_cons_abun_prod_split, 
                                              df_speciesMetab=df_speciesMetab, 
                                              num_passages=6, num_bioRep=3, 
                                              sample_names=list(df_speciesAbun_split.columns.values))
    return sample_names_split, mat_cons_abun_split_list

def iterate_growth_ratio_pvec(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, p_vec):
    df_speciesAbun_split_dict = {}
    for i, p_ in enumerate(p_vec):
        df_speciesAbun_split_dict[i] = geometric_avg(df_speciesAbun_prev, df_speciesAbun_next, p=p_)
    
    df_cons_abun_prod_split = np.zeros((df_speciesAbun_prev.shape[1], df_speciesMetab.shape[1]))
    for row_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_split = np.zeros((df_speciesAbun_prev.shape[0], df_speciesMetab.shape[1]))
        for i, p_ in enumerate(p_vec):
            df_speciesAbun_split[:, i] = df_speciesAbun_split_dict[i].iloc[:, row_]
        df_cons_abun_prod_split[row_, :] = \
            compute_cons_abun_prod_new(df_speciesMetab=df_speciesMetab, \
                                       df_speciesAbun=df_speciesAbun_split)
    sample_names_split, mat_cons_abun_split_list = \
        compute_species_metab_matrix_for_nnls(df_cons_abun_prod=df_cons_abun_prod_split, 
                                              df_speciesMetab=df_speciesMetab, 
                                              num_passages=6, num_bioRep=3, 
                                              sample_names=list(df_speciesAbun_prev.columns.values))
    return sample_names_split, mat_cons_abun_split_list

def get_metab_names_orderedRi(Ri, df_metabNames, df_metabIds):
    metab_names_all = df_metabNames.iloc[:, 1].values
    metab_id_data = df_metabIds.iloc[:, 0].values - 1
    metab_names = metab_names_all[metab_id_data]
    
    id_order = np.argsort(-Ri)
    Ri_ordered = Ri.copy()[id_order]
    metab_names_ordered = metab_names[id_order]
    
    return id_order, Ri_ordered, metab_names_ordered

def create_abundance_header(num_bioRep=3, num_passages=6):
    '''Create header for the datafram with species abundance'''
    header_ = []
    for pass_ in range(num_passages):
        for rep_ in range(num_bioRep):
            header_.append('p' + str(pass_ + 1) + "_" + 'r' + str(rep_))
    return header_

def create_abundance_header_new(num_bioRep=3, num_passages=6):
    '''Create header for the datafram with species abundance'''
    header_ = []
    for rep_ in range(num_bioRep):
        for pass_ in range(num_passages):
            header_.append('p' + str(pass_ + 1) + "_" + 'r' + str(rep_))
    return header_

def expand_passage_by_rep(passage_='p1', num_bioRep=3):
    '''Expand passage name to include replicate numbers as well'''
    passage_rep = []
    for rep_ in range(num_bioRep):
        passage_rep.append(passage_ + "_" + "r" + str(rep_))
        
    return passage_rep

def compute_cons_abun_prod(df_speciesMetab, df_speciesAbun, power_=1.0):
    '''Compute matrix product of species-metabolite consumption matrix and species abundance matrix'''        
    df_speciesAbun_T = df_speciesAbun.copy()**power_
    df_speciesAbun_T = df_speciesAbun_T.transpose()
#     df_cons_abun_prod = df_speciesAbun_T.dot(df_speciesMetab)
    df_cons_abun_prod = np.matmul(np.array(df_speciesAbun_T.values), np.array(df_speciesMetab.values))
    return df_cons_abun_prod

def compute_cons_abun_prod_new(df_speciesMetab, df_speciesAbun):
    '''Compute matrix product of species-metabolite consumption matrix and species abundance matrix'''
    df_speciesAbun_T = df_speciesAbun.copy()
    df_speciesAbun_T = df_speciesAbun_T.transpose()
#     df_cons_abun_prod = df_speciesAbun_T.dot(df_speciesMetab)
    df_cons_abun_prod = \
        np.multiply(np.array(df_speciesAbun), np.array(df_speciesMetab.values)).sum(axis=0)
    return df_cons_abun_prod

def compute_species_metab_matrix_for_nnls(df_cons_abun_prod, df_speciesMetab, num_passages=6, num_bioRep=3, 
                                          sample_names=None):
    '''create one matrix of c_i^alpha / (sum_beta c_i^beta B_beta(k)) (species X metabolites) 
       for each passage and bioreplicate'''
    num_species_tmp = df_speciesMetab.shape[0]
    num_metabs_tmp = df_speciesMetab.shape[1]
    if sample_names is None:
        sample_names = create_abundance_header(num_passages=6, num_bioRep=3)
    mat_cons_abun_list = {}
    tmp_ones = np.ones((num_species_tmp, 1))
    for sample_ in sample_names:
#         df_denom = df_cons_abun_prod.loc[sample_, :]
        df_denom = df_cons_abun_prod[sample_names.index(sample_), :]
        tmp_mat = np.array(df_denom).reshape((1, num_metabs_tmp))
        id_ = np.where(tmp_mat.flatten() == 0)[0]
        if len(id_) > 0:
            print(id_)
        mat_denom = np.matmul(tmp_ones, tmp_mat)
        mat_cons_abun_list[sample_] = \
            np.divide(np.array(df_speciesMetab.copy().values), mat_denom)
        
    return sample_names, mat_cons_abun_list


def null_Ri_2(df_speciesMetab, df_cons_abun_prod_split):
    tmp = df_speciesMetab.sum(axis=1)
    tmp = np.array(df_speciesMetab.sum(axis=1)).mean()
    tmp = df_cons_abun_prod_split.mean(axis=0) / tmp
    return tmp


def get_rank(arr):
    # Get the sorted indices
    sorted_indices = np.argsort(arr)

    # Initialize an array to hold the ranks
    ranks = np.zeros_like(sorted_indices)

    # Rank the elements
    ranks[sorted_indices] = np.arange(len(arr))

    # Calculate the average rank for repeated values
    unique_arr, unique_counts = np.unique(arr, return_counts=True)
    repeat_indices = np.where(unique_counts > 1)[0]
    for idx in repeat_indices:
        repeat_mask = (arr == unique_arr[idx])
        repeat_ranks = ranks[repeat_mask]
        mean_rank = repeat_ranks.mean()
        ranks[repeat_mask] = mean_rank
    return ranks


def geometric_avg(mat_init, mat_final, p=0):
    mat_split = mat_init**p * mat_final**(1 - p)
    return mat_split


def post_process_hdbscan_clusters(cluster_labels):
    clust_labels, clust_counts = np.unique(cluster_labels, return_counts=True)
    cluster_labels_new = np.zeros((len(cluster_labels)))

    for label_ in clust_labels:
        if label_ >= 0:
            cluster_labels_new[cluster_labels == label_] = label_
        else:
            len_labels = len(cluster_labels[cluster_labels == label_])
            cluster_labels_new[cluster_labels == label_] = \
                np.arange(len(clust_labels) - 1, len(clust_labels) - 1 + len_labels)
    return cluster_labels_new.astype(int)


def remove_passages(pass_rm_list, num_passages=6, num_brep=3):
    pass_ex_list = []
    for pass_ in pass_rm_list:
        for rep_ in range(num_brep):
            pass_ex_list += [pass_ + rep_ * num_passages]
    pass_keep = list(set(range(num_passages * num_brep)) - set(pass_ex_list))
    return pass_keep


def compute_Ri_ss_goodness_of_fit(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, \
                                  df_speciesAbun_ratio, \
                                  id_species, Ri_fit, p_tmp=0):
    
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    lhs_ = []
    rhs_ = []
    sample_rm = []
    species_rm = []
    for sample_ in sample_names_split:
        pass_ = int(sample_.split("_")[0][1])
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=0)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
            sample_id = sample_names_split.index(sample_)
            abun_tmp = \
                df_speciesAbun_prev_tmp.iloc[id_species, sample_id].values.reshape((len(id_species), 1))
            b_train_sample = abun_tmp.flatten()
            abun_tmp = np.matmul(abun_tmp, np.ones((1, df_speciesMetab.shape[1])))
            A_train_sample = np.multiply(A_train_sample, abun_tmp)

            lhs_ = np.matmul(A_train_sample, Ri_fit).flatten()
            id_rm = np.where(lhs_ > 1)[0]
            sample_rm += [sample_] * len(id_rm)
            species_rm += list(id_rm)
            
            
            if count_ == 0:
                A_train = A_train_sample
                b_train = b_train_sample
                count_ += 1
            else:
                A_train = np.vstack((A_train, A_train_sample))
                b_train = np.hstack((b_train, b_train_sample))
#     b_train = np.ones((A_train.shape[0]))
    lhs_ = b_train.flatten()
    rhs_ = np.matmul(A_train, Ri_fit).flatten()
    return lhs_, rhs_, sample_rm, species_rm

def compute_Ri_dynamic_goodness_of_fit(df_speciesMetab, df_speciesAbun_prev, \
                                       df_speciesAbun_next, df_speciesAbun_ratio, \
                                       id_species, Ri_fit, p_tmp):
    
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    df_speciesAbun_ratio_new = df_speciesAbun_ratio.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        # pass_ = int(sample_.split("_")[0][1])
        pass_ = 0
        
        p_tmp_new = p_tmp
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=p_tmp_new)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
            sample_id = sample_names_split.index(sample_)
            tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            tmp_ratio = tmp.copy()
            id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                                (df_speciesAbun_next[sample_].values == 1e-8))[0]
            id_keep = list(set(id_species) - set(id_notkeep))
    #         tmp = df_speciesAbun_ratio_tmp_[sample_].values
            tmp = np.array(tmp).reshape((len(id_species), 1))
            tmp = tmp[id_keep, :]
            vals_ = np.ones((len(id_species)))
            vals_tmp = np.multiply((np.matmul(A_train_sample, Ri_fit).flatten())[id_keep], \
                                tmp.flatten()**(1 - p_tmp))
            vals_[id_keep] = vals_tmp.copy()

            if count_ == 0:
                rhs_ = vals_.flatten()[id_keep]
                lhs_ = tmp_ratio.flatten()[id_keep]
                count_ += 1
            else:
                rhs_ = np.hstack([rhs_, vals_.flatten()[id_keep]])
                lhs_ = np.hstack([lhs_, tmp_ratio.flatten()[id_keep]])

    return lhs_, rhs_, None, None

def compute_Ri_hybrid(df_speciesMetab, df_speciesAbun_, df_speciesAbun_prev, \
                      df_speciesAbun_next, df_speciesAbun_ratio, num_passages, \
                      p_tmp, id_species):
    # steady state part
    df_speciesAbun_prev_tmp = df_speciesAbun_.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_.copy()
    num_species_tmp = df_speciesAbun_.shape[0]

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        pass_ = int(sample_.split("_")[0][1])
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=0)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
            sample_id = sample_names_split.index(sample_)
            if count_ == 0:
                A_train = A_train_sample
                count_ += 1
            else:
                A_train = np.vstack((A_train, A_train_sample))
                
      
    # dynamical part
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    df_speciesAbun_ratio_new = df_speciesAbun_ratio.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        pass_ = int(sample_.split("_")[0][1])
        
        p_tmp_new = p_tmp
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=p_tmp_new)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
            sample_id = sample_names_split.index(sample_)
            tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            tmp1 = np.array(df_speciesAbun_ratio.iloc[id_species, sample_id])
            id_keep = np.where(tmp1 != -1)[0]
            id_not_keep = np.where(tmp1 == -1)[0]
            tmp[id_not_keep] = 1
            tmp_ratio = tmp.copy()
            tmp = np.array(tmp**(-p_tmp_new)).reshape((len(id_species), 1))
#             tmp = np.array(tmp).reshape((len(train_strain_id), 1))
#             tmp = tmp[id_keep, :]
            tmp = np.matmul(tmp, np.ones((1, A_train_sample.shape[1])))
            A_train_sample = np.multiply(A_train_sample, tmp)
            A_train_sample = A_train_sample[id_keep, :]
            
            A_train = np.vstack((A_train, A_train_sample))
                
    b_train = np.ones((A_train.shape[0]))
    coeff_train_ = nnls(A_train, b_train.flatten())[0]
    return coeff_train_


def compute_Ri_ss(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, \
                  p_tmp, id_species):
    
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        pass_ = int(sample_.split("_")[0][1])
        
        p_tmp_new = p_tmp
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=p_tmp_new)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
            sample_id = sample_names_split.index(sample_)
            if count_ == 0:
                A_train = A_train_sample
                count_ += 1
            else:
                A_train = np.vstack((A_train, A_train_sample))
    b_train = np.ones((A_train.shape[0]))
    coeff_train_ = nnls(A_train, b_train.flatten())[0]
    return coeff_train_

def weighted_avg_consumption(A_train_sample, metabs_cluster_id):
    num_clusters = len(metabs_cluster_id)
    A_train_sample_new = np.zeros((A_train_sample.shape[0], num_clusters))
    for count_, metabs_id_ in enumerate(metabs_cluster_id):
        A_train_sample_new[:, count_] = \
            np.sum(A_train_sample[:, metabs_id_], axis=1)
    return A_train_sample_new

def get_matrix_form_prod(prod_mat, A_mat, Ri, N_yk):
    mat_2 = np.zeros((A_mat.shape[0], prod_mat.shape[0]))
    for m in range(A_mat.shape[0]):
        tmp = Ri.copy() * A_mat[m, :].flatten()
        tmp = tmp.reshape((-1, 1))
        mat_2[m, :] = (prod_mat @ tmp).flatten() * N_yk.flatten()

    return mat_2


def compute_Ri_no_f(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, \
               df_speciesAbun_ratio, \
               p_tmp, num_passages, id_species, method="linear", alpha=0, \
               df_speciesAbun_ratio_nonoise=None, \
               metabs_cluster_id=None, get_prod=False, B_alone=None, \
               df_speciesMetab_prod=None, prod_use_prev=True, \
               use_dilution_term=False, dilution_factor=15000, \
               use_avg_for_prod=True, check_ratio_dir=True, mode_Ri=True, Ri=None, \
               power_=1.0):
    
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio.copy()
    if df_speciesAbun_ratio_nonoise is not None:
        df_speciesAbun_ratio_new = df_speciesAbun_ratio_nonoise.copy()
    else:
        df_speciesAbun_ratio_new = df_speciesAbun_ratio.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    df_speciesAbun_split = \
        geometric_avg(df_speciesAbun_prev_tmp, df_speciesAbun_next_tmp, p=p_tmp)
    if prod_use_prev:
        df_split = df_speciesAbun_prev_tmp.copy()
    else:
        df_split = geometric_avg(df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p=p_tmp)
    if use_avg_for_prod:
        df_tmp = df_split.copy()
    else:
        df_tmp = df_speciesAbun_next_tmp.copy()
    prod_metabs_cond = \
        get_prod_term(df_speciesMetab_prod, df_tmp.copy(), B_alone, \
                      get_prod=get_prod)

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        # pass_ = int(sample_.split("_")[0][1])
        pass_ = 2
        
        p_tmp_new = p_tmp
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio_2_0(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=p_tmp_new, \
                                 df_speciesAbun_ratio=df_speciesAbun_ratio_tmp, power_=power_)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :] * \
                prod_metabs_cond[sample_][id_species, :]
            if metabs_cluster_id is not None:
                A_train_sample = \
                    weighted_avg_consumption(A_train_sample, metabs_cluster_id)
            sample_id = sample_names_split.index(sample_)
            df_avg = np.array(df_speciesAbun_next_tmp.iloc[id_species, sample_id] / \
                (np.log(df_speciesAbun_next_tmp.iloc[id_species, sample_id]) + \
                 np.log(dilution_factor) - \
                 np.log(df_speciesAbun_prev_tmp.iloc[id_species, sample_id]))).reshape((-1, 1))
            tmp = np.array(df_speciesAbun_ratio_tmp.iloc[id_species, sample_id])
            tmp1 = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            id_keep = np.where(tmp1 > 0)[0]
            id_not_keep = np.where(tmp1 < 0)[0]
            tmp[id_not_keep] = 1
            tmp = np.array(tmp**(-p_tmp_new)).reshape((len(id_species), 1))
            tmp_ratio = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            tmp_find = tmp1.copy()
            tmp_ratio = tmp_ratio**(p_tmp_new)
            
#             tmp = np.array(tmp).reshape((len(train_strain_id), 1))
#             tmp = tmp[id_keep, :]
            tmp = np.matmul(df_avg, np.ones((1, A_train_sample.shape[1])))

            # if check_ratio_dir:
            #     id_increase = np.where(tmp_find >= 1)[0]
            #     id_decrease = np.where(tmp_find < 1)[0]
            # else:
            #     id_increase = np.where(tmp_find > 0)[0]
            #     id_decrease = np.where(tmp_find < 0)[0]
            A_train_sample[:, :] = \
                np.multiply(A_train_sample[:, :], \
                            tmp[:, :])
            # if use_dilution_term:
            #     tmp_dil = \
            #         np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id]).flatten()
            #     tmp_dil[id_decrease] = tmp_dil[id_decrease]**(-1)
            #     tmp_dil[id_increase] = \
            #         tmp_dil[id_increase]**(p_tmp_new-1)
            #     # tmp_dil = tmp_dil.reshape((-1, 1))
            #     tmp_dil /= dilution_factor
            # A_train_sample[id_increase, :] = \
            #     np.multiply(A_train_sample[id_increase, :], \
            #                 tmp[id_increase, :])
            A_train_sample = A_train_sample[id_keep, :]

            # b_val = np.ones((len(id_species)))
            b_val = np.array(df_speciesAbun_next_tmp.iloc[id_species, sample_id])
            b_val = b_val[id_keep]


            if not mode_Ri:
                intercept_ = A_train_sample.copy() @ Ri
                b_val -= intercept_.flatten()
                mat_new = get_matrix_form_prod(np.array(df_speciesMetab_prod), \
                                               np.array(A_train_sample), Ri, \
                                               np.array(df_tmp.iloc[:, sample_id]))
                A_train_sample = mat_new.copy()

            if count_ == 0:
                # if use_dilution_term:
                #     A_train_sample = np.hstack((A_train_sample, tmp_dil[id_keep, :]))
                A_train = A_train_sample
                # b_train = tmp_ratio[id_keep].flatten()
                b_train = b_val.flatten()
                count_ += 1
            else:
                # if use_dilution_term:
                #     A_train_sample = np.hstack((A_train_sample, tmp_dil[id_keep, :]))
                A_train = np.vstack((A_train, A_train_sample))
                # b_train = np.hstack((b_train, tmp_ratio[id_keep]))
                b_train = np.hstack((b_train, b_val.flatten()))
    # b_train = np.ones((A_train.shape[0]))
    # print(np.min(A_train.flatten()))

    if method == "linear":
        coeff_train_ = nnls(A_train, b_train.flatten())[0]
    else:
        regr = \
            ElasticNet(random_state=0, alpha=alpha, l1_ratio=1, \
                       fit_intercept=False, positive=True)
        regr.fit(A_train, b_train.flatten())
        coeff_train_ = regr.coef_
    # coeff_train_ = nnls(np.log10(A_train), np.log10(b_train.flatten()))[0]
    return coeff_train_, A_train, b_train


def compute_Ri(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, \
               df_speciesAbun_ratio, \
               p_tmp, num_passages, id_species, method="linear", alpha=0, \
               df_speciesAbun_ratio_nonoise=None, \
               metabs_cluster_id=None, get_prod=False, B_alone=None, \
               df_speciesMetab_prod=None, prod_use_prev=True, \
               use_dilution_term=False, dilution_factor=15000, \
               use_avg_for_prod=True, check_ratio_dir=True, mode_Ri=True, Ri=None, \
               power_=1.0):
    
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio.copy()
    if df_speciesAbun_ratio_nonoise is not None:
        df_speciesAbun_ratio_new = df_speciesAbun_ratio_nonoise.copy()
    else:
        df_speciesAbun_ratio_new = df_speciesAbun_ratio.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    df_speciesAbun_split = \
        geometric_avg(df_speciesAbun_prev_tmp, df_speciesAbun_next_tmp, p=p_tmp)
    if prod_use_prev:
        df_split = df_speciesAbun_prev_tmp.copy()
    else:
        df_split = geometric_avg(df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p=p_tmp)
    if use_avg_for_prod:
        df_tmp = df_split.copy()
    else:
        df_tmp = df_speciesAbun_next_tmp.copy()
    prod_metabs_cond = \
        get_prod_term(df_speciesMetab_prod, df_tmp.copy(), B_alone, \
                      get_prod=get_prod)

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        # pass_ = int(sample_.split("_")[0][1])
        pass_ = 2
        
        p_tmp_new = p_tmp
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=p_tmp_new, \
                                 df_speciesAbun_ratio=df_speciesAbun_ratio_tmp, power_=power_)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :] * \
                prod_metabs_cond[sample_][id_species, :]
            if metabs_cluster_id is not None:
                A_train_sample = \
                    weighted_avg_consumption(A_train_sample, metabs_cluster_id)
            sample_id = sample_names_split.index(sample_)
            tmp = np.array(df_speciesAbun_ratio_tmp.iloc[id_species, sample_id])
            tmp1 = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            id_keep = np.where(tmp1 > 0)[0]
            id_not_keep = np.where(tmp1 < 0)[0]
            tmp[id_not_keep] = 1
            tmp = np.array(tmp**(-p_tmp_new)).reshape((len(id_species), 1))
            tmp_ratio = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            tmp_find = tmp1.copy()
            tmp_ratio = tmp_ratio**(p_tmp_new)
            
#             tmp = np.array(tmp).reshape((len(train_strain_id), 1))
#             tmp = tmp[id_keep, :]
            tmp = np.matmul(tmp, np.ones((1, A_train_sample.shape[1])))

            if check_ratio_dir:
                id_increase = np.where(tmp_find >= 1)[0]
                id_decrease = np.where(tmp_find < 1)[0]
            else:
                id_increase = np.where(tmp_find > 0)[0]
                id_decrease = np.where(tmp_find < 0)[0]
            A_train_sample[id_decrease, :] = \
                np.multiply(A_train_sample[id_decrease, :], \
                            tmp[id_decrease, :])
            if use_dilution_term:
                tmp_dil = \
                    np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id]).flatten()
                tmp_dil[id_decrease] = tmp_dil[id_decrease]**(-1)
                tmp_dil[id_increase] = \
                    tmp_dil[id_increase]**(p_tmp_new-1)
                # tmp_dil = tmp_dil.reshape((-1, 1))
                tmp_dil /= dilution_factor
            # A_train_sample[id_increase, :] = \
            #     np.multiply(A_train_sample[id_increase, :], \
            #                 tmp[id_increase, :])
            A_train_sample = A_train_sample[id_keep, :]

            b_val = np.ones((len(id_species)))
            # b_val[id_increase] = tmp_ratio[id_increase]
            # b_val = tmp_ratio
            b_val[id_increase] = tmp_ratio[id_increase]
            # b_val[id_decrease] = tmp_ratio[id_decrease] / tmp[id_decrease, 0].flatten()
            b_val = b_val[id_keep]

            if use_dilution_term:
                b_val -= tmp_dil[id_keep].flatten()

            if not mode_Ri:
                intercept_ = A_train_sample.copy() @ Ri
                b_val -= intercept_.flatten()
                mat_new = get_matrix_form_prod(np.array(df_speciesMetab_prod), \
                                               np.array(A_train_sample), Ri, \
                                               np.array(df_tmp.iloc[:, sample_id]))
                A_train_sample = mat_new.copy()

            if count_ == 0:
                # if use_dilution_term:
                #     A_train_sample = np.hstack((A_train_sample, tmp_dil[id_keep, :]))
                A_train = A_train_sample
                # b_train = tmp_ratio[id_keep].flatten()
                b_train = b_val.flatten()
                count_ += 1
            else:
                # if use_dilution_term:
                #     A_train_sample = np.hstack((A_train_sample, tmp_dil[id_keep, :]))
                A_train = np.vstack((A_train, A_train_sample))
                # b_train = np.hstack((b_train, tmp_ratio[id_keep]))
                b_train = np.hstack((b_train, b_val.flatten()))
    # b_train = np.ones((A_train.shape[0]))
    # print(np.min(A_train.flatten()))

    if method == "linear":
        coeff_train_ = nnls(A_train, b_train.flatten())[0]
    else:
        regr = \
            ElasticNet(random_state=0, alpha=alpha, l1_ratio=1, \
                       fit_intercept=False, positive=True)
        regr.fit(A_train, b_train.flatten())
        coeff_train_ = regr.coef_
    # coeff_train_ = nnls(np.log10(A_train), np.log10(b_train.flatten()))[0]
    return coeff_train_, A_train, b_train

def compute_Ri_tmp(df_speciesMetab, df_speciesAbun_prev, df_speciesAbun_next, \
               df_speciesAbun_ratio, \
               p_tmp, num_passages, id_species, method="linear", alpha=0):
    
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    df_speciesAbun_ratio_new = df_speciesAbun_ratio.copy()
    num_species_tmp = df_speciesAbun_prev.shape[0]

    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    count_ = 0 
    for sample_ in sample_names_split:
        # pass_ = int(sample_.split("_")[0][1])
        pass_ = 2
        
        p_tmp_new = p_tmp
        
        _, mat_cons_abun_split_list_tmp = \
            iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                 df_speciesAbun_next_tmp.copy(), p_=p_tmp_new)
        if pass_ > -1:
            brep_ = int(sample_.split("_")[1][1])
            A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
            sample_id = sample_names_split.index(sample_)
            tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
            tmp1 = np.array(df_speciesAbun_ratio.iloc[id_species, sample_id])
            id_keep = np.where(tmp1 != -1)[0]
            id_not_keep = np.where(tmp1 == -1)[0]
            tmp[id_not_keep] = 1
            tmp_ratio = tmp.copy()
            tmp_find = tmp.copy()
            tmp_ratio = tmp_ratio**(p_tmp_new)
            tmp = np.array(tmp**(-p_tmp_new)).reshape((len(id_species), 1))
#             tmp = np.array(tmp).reshape((len(train_strain_id), 1))
#             tmp = tmp[id_keep, :]
            tmp = np.matmul(tmp, np.ones((1, A_train_sample.shape[1])))

            id_increase = np.where(tmp_find >= 1)[0]
            id_decrease = np.where(tmp_find < 1)[0]
            A_train_sample[id_decrease, :] = \
                np.multiply(A_train_sample[id_decrease, :], tmp[id_decrease, :])
            A_train_sample = A_train_sample[id_keep, :]

            b_val = np.ones((len(id_species)))
            b_val[id_increase] = tmp_ratio[id_increase]
            b_val = b_val[id_keep]

            if count_ == 0:
                A_train = A_train_sample
                # b_train = tmp_ratio[id_keep].flatten()
                b_train = b_val.flatten()
                count_ += 1
            else:
                A_train = np.vstack((A_train, A_train_sample))
                # b_train = np.hstack((b_train, tmp_ratio[id_keep]))
                b_train = np.hstack((b_train, b_val.flatten()))
    # b_train = np.ones((A_train.shape[0]))
    if method == "linear":
        coeff_train_ = nnls(A_train, b_train.flatten())[0]
    else:
        regr = \
            ElasticNet(random_state=0, alpha=alpha, l1_ratio=1, \
                       fit_intercept=False, positive=True)
        regr.fit(A_train, b_train.flatten())
        coeff_train_ = regr.coef_
    # coeff_train_ = nnls(np.log10(A_train), np.log10(b_train.flatten()))[0]
    return coeff_train_


def compute_Ri_bal(df_speciesMetab, df_speciesAbun_prev, \
                   df_speciesAbun_next, df_speciesAbun_ratio, \
                   p_tmp, num_passages, num_rand=100):
    num_metabs = df_speciesMetab.shape[1]
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy()
    df_speciesAbun_ratio_new = df_speciesAbun_ratio.copy()
    sample_names_split = list(df_speciesAbun_prev_tmp.columns.values)
    num_species_tmp = df_speciesAbun_prev.shape[0]
    Ri_fit = np.zeros((num_metabs))
    count_rand = 0
    for rand_ in range(num_rand):
        print(f'rand loop = {rand_}')
        count_ = 0 
        for sample_ in sample_names_split:
            # pass_ = int(sample_.split("_")[0][1])
            sample_id = sample_names_split.index(sample_)
            pass_ = 2
            
            p_tmp_new = p_tmp

            ratio_ = df_speciesAbun_ratio_new.iloc[:, sample_id].values
            id_increase = np.where(ratio_ >= 1)[0]
            id_decrease = np.where(ratio_ < 1)[0]

            if len(id_decrease) > len(id_increase):
                id_species = id_increase
                id_sel = np.random.choice(id_decrease, len(id_increase), replace=False)
                id_species = np.hstack([id_species, id_sel])
            else:
                id_species = id_decrease
                id_sel = np.random.choice(id_increase, len(id_decrease), replace=False)
                id_species = np.hstack([id_species, id_sel])
            
            _, mat_cons_abun_split_list_tmp = \
                iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev_tmp.copy(), \
                                    df_speciesAbun_next_tmp.copy(), p_=p_tmp_new)
            if pass_ > -1:
                brep_ = int(sample_.split("_")[1][1])
                A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
                tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
                tmp1 = np.array(df_speciesAbun_ratio.iloc[id_species, sample_id])
                id_keep = np.where(tmp1 != -1)[0]
                id_not_keep = np.where(tmp1 == -1)[0]
                tmp[id_not_keep] = 1
                tmp_ratio = tmp.copy()
                tmp_find = tmp.copy()
                tmp_ratio = tmp_ratio**(p_tmp_new)
                tmp = np.array(tmp**(-p_tmp_new)).reshape((len(id_species), 1))
    #             tmp = np.array(tmp).reshape((len(train_strain_id), 1))
    #             tmp = tmp[id_keep, :]
                tmp = np.matmul(tmp, np.ones((1, A_train_sample.shape[1])))

                id_increase = np.where(tmp_find >= 1)[0]
                id_decrease = np.where(tmp_find < 1)[0]
                A_train_sample[id_decrease, :] = \
                    np.multiply(A_train_sample[id_decrease, :], tmp[id_decrease, :])
                A_train_sample = A_train_sample[id_keep, :]

                b_val = np.ones((len(id_species)))
                b_val[id_increase] = tmp_ratio[id_increase]
                b_val = b_val[id_keep]

                if count_ == 0:
                    A_train = A_train_sample
                    # b_train = tmp_ratio[id_keep].flatten()
                    b_train = b_val.flatten()
                    count_ += 1
                else:
                    A_train = np.vstack((A_train, A_train_sample))
                    # b_train = np.hstack((b_train, tmp_ratio[id_keep]))
                    b_train = np.hstack((b_train, b_val.flatten()))
        # b_train = np.ones((A_train.shape[0]))
        coeff_train_ = nnls(A_train, b_train.flatten())[0]
    # coeff_train_ = nnls(np.log10(A_train), np.log10(b_train.flatten()))[0]

        if np.sum(coeff_train_) <= 1.5:
            Ri_fit += coeff_train_
            count_rand += 1
    Ri_fit /= count_rand
    return Ri_fit


def compute_growth_ratio_iterate_blind_no_f(df_speciesAbun_prev, df_speciesAbun_next, p_tmp, Ri, growth_ratios_, \
                                       ratio_means_, \
                                       df_speciesMetab, norm_=True, \
                                       metabs_cluster_id=None, \
                                       get_prod=False, B_alone=None, \
                                       df_speciesMetab_prod=None, \
                                       prod_use_prev=True, \
                                       use_dilution=False, \
                                       dilution_factor=15000, \
                                       use_avg_for_prod=True):
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = growth_ratios_.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (growth_ratios_.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
    sample_names_split, mat_cons_abun_split_list_tmp = \
        iterate_growth_ratio_2_0(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                             df_speciesAbun_next.copy(), p_=p_tmp)
    if prod_use_prev:
        df_split = df_speciesAbun_prev.copy()
    else:
        df_split = geometric_avg(df_speciesAbun_prev.copy(), \
                                 df_speciesAbun_next.copy(), p=p_tmp)
    if use_avg_for_prod:
        df_tmp = df_split.copy()
    else:
        df_tmp = df_speciesAbun_next.copy()
    prod_metabs_cond = \
        get_prod_term(df_speciesMetab_prod, df_tmp.copy(), B_alone, \
                      get_prod=get_prod)

    id_species = range(df_speciesAbun_prev.shape[0])
    count_ = 0
    df_growth_ratio = pd.DataFrame()
#     ratio_fac = 0
#     vals_ = {}
    for sample_ in sample_names_split:
#         pass_ = int(sample_.split("_")[0][1])
#         brep_ = int(sample_.split("_")[1][1])
        A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
        if metabs_cluster_id is not None:
            A_train_sample = \
                weighted_avg_consumption(A_train_sample, metabs_cluster_id)
        A_train_sample = A_train_sample * prod_metabs_cond[sample_][id_species, :]
        sample_id = sample_names_split.index(sample_)
        tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
        id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                              (df_speciesAbun_next[sample_].values == 1e-8))[0]
        id_keep = list(set(id_species) - set(id_notkeep))
#         tmp = df_speciesAbun_ratio_tmp_[sample_].values

        df_avg = np.array(df_speciesAbun_next.iloc[id_species, sample_id] / \
            (np.log(df_speciesAbun_next.iloc[id_species, sample_id]) + \
                np.log(dilution_factor) - \
                np.log(df_speciesAbun_prev.iloc[id_species, sample_id]))).reshape((-1, 1))
        tmp = np.matmul(df_avg, np.ones((1, A_train_sample.shape[1])))

        A_train_sample[:, :] = \
            np.multiply(A_train_sample[:, :], \
                        tmp[:, :])

        # tmp = np.array(tmp).reshape((len(id_species), 1))
        # tmp = tmp[id_keep, :]
        vals_ = np.ones((len(id_species)))
        vals_tmp = (np.matmul(A_train_sample, Ri).flatten())[id_keep]
        vals_[id_keep] = vals_tmp.copy()
        if use_dilution:
            vals_[id_keep] += 1 / dilution_factor
        else:
            vals_[vals_ == 0] = 1 / dilution_factor
#         vals_ = df_speciesAbun_ratio_tmp_[sample_].values
#         vals_[sample_] = (np.matmul(A_train_sample, Ri).flatten())**(1 / p_tmp)

        if norm_:
#             mean_ = 10 ** np.mean(np.log10(vals_[id_keep]))
#             ratio_fac = 10 ** (ratio_means_[sample_] - np.mean(np.log10(vals_[id_keep])))
#             vals_[id_keep] = np.power(10, (np.log10(vals_[id_keep]) * ratio_means_[sample_] / \
#                                     np.mean(np.log10(vals_[id_keep]))))
#     #         vals_ += 1 - np.sum(vals_ * df_speciesAbun_prev[sample_].values)
            vals_next = vals_.copy() * df_speciesAbun_prev[sample_].values
            ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_ * ratio_fac
    
            # ratio_fac = ratio_means_[sample_] / np.sum(np.log10(vals_[id_keep]))
            # vals_ = 10**(np.log10(vals_) * ratio_fac)
            # df_growth_ratio[sample_] = vals_
        else:
#             vals_next = vals_ * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) * ratio_means_[sample_] / \
#                                           np.mean(np.log10(vals_)))
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) + ratio_means_[sample_] - \
#                                           np.mean(np.log10(vals_)))
    
#     growth_ratios_new = np.zeros(())
    
    return df_growth_ratio

def compute_growth_ratio_iterate_blind(df_speciesAbun_prev, df_speciesAbun_next, p_tmp, Ri, growth_ratios_, \
                                       ratio_means_, \
                                       df_speciesMetab, norm_=True, \
                                       metabs_cluster_id=None, \
                                       get_prod=False, B_alone=None, \
                                       df_speciesMetab_prod=None, \
                                       prod_use_prev=True, \
                                       use_dilution=False, \
                                       dilution_factor=15000, \
                                       use_avg_for_prod=True):
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = growth_ratios_.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (growth_ratios_.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
    sample_names_split, mat_cons_abun_split_list_tmp = \
        iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                             df_speciesAbun_next.copy(), p_=p_tmp)
    if prod_use_prev:
        df_split = df_speciesAbun_prev.copy()
    else:
        df_split = geometric_avg(df_speciesAbun_prev.copy(), \
                                 df_speciesAbun_next.copy(), p=p_tmp)
    if use_avg_for_prod:
        df_tmp = df_split.copy()
    else:
        df_tmp = df_speciesAbun_next.copy()
    prod_metabs_cond = \
        get_prod_term(df_speciesMetab_prod, df_tmp.copy(), B_alone, \
                      get_prod=get_prod)

    id_species = range(df_speciesAbun_prev.shape[0])
    count_ = 0
    df_growth_ratio = pd.DataFrame()
#     ratio_fac = 0
#     vals_ = {}
    for sample_ in sample_names_split:
#         pass_ = int(sample_.split("_")[0][1])
#         brep_ = int(sample_.split("_")[1][1])
        A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
        if metabs_cluster_id is not None:
            A_train_sample = \
                weighted_avg_consumption(A_train_sample, metabs_cluster_id)
        A_train_sample = A_train_sample * prod_metabs_cond[sample_][id_species, :]
        sample_id = sample_names_split.index(sample_)
        tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
        id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                              (df_speciesAbun_next[sample_].values == 1e-8))[0]
        id_keep = list(set(id_species) - set(id_notkeep))
#         tmp = df_speciesAbun_ratio_tmp_[sample_].values
        tmp = np.array(tmp).reshape((len(id_species), 1))
        tmp = tmp[id_keep, :]
        vals_ = np.ones((len(id_species)))
        vals_tmp = np.multiply((np.matmul(A_train_sample, Ri).flatten())[id_keep], \
                               tmp.flatten()**(1 - p_tmp))
        vals_[id_keep] = vals_tmp.copy()
        if use_dilution:
            vals_[id_keep] += 1 / dilution_factor
        else:
            vals_[vals_ == 0] = 1 / dilution_factor
#         vals_ = df_speciesAbun_ratio_tmp_[sample_].values
#         vals_[sample_] = (np.matmul(A_train_sample, Ri).flatten())**(1 / p_tmp)

        if norm_:
#             mean_ = 10 ** np.mean(np.log10(vals_[id_keep]))
#             ratio_fac = 10 ** (ratio_means_[sample_] - np.mean(np.log10(vals_[id_keep])))
#             vals_[id_keep] = np.power(10, (np.log10(vals_[id_keep]) * ratio_means_[sample_] / \
#                                     np.mean(np.log10(vals_[id_keep]))))
#     #         vals_ += 1 - np.sum(vals_ * df_speciesAbun_prev[sample_].values)
            vals_next = vals_.copy() * df_speciesAbun_prev[sample_].values
            ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_ * ratio_fac
    
            # ratio_fac = ratio_means_[sample_] / np.sum(np.log10(vals_[id_keep]))
            # vals_ = 10**(np.log10(vals_) * ratio_fac)
            # df_growth_ratio[sample_] = vals_
        else:
#             vals_next = vals_ * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) * ratio_means_[sample_] / \
#                                           np.mean(np.log10(vals_)))
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) + ratio_means_[sample_] - \
#                                           np.mean(np.log10(vals_)))
    
#     growth_ratios_new = np.zeros(())
    
    return df_growth_ratio

def compute_log_growth_ratio_iterate_blind(df_speciesAbun_prev, \
                                           df_speciesAbun_next, p_tmp, Ri, growth_ratios_, \
                                           ratio_means_, df_speciesMetab, norm_=True):
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = 10**growth_ratios_.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (10**growth_ratios_.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
    sample_names_split, mat_cons_abun_split_list_tmp = \
        iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                             df_speciesAbun_next.copy(), p_=p_tmp)

    id_species = range(df_speciesAbun_prev.shape[0])
    count_ = 0
    df_growth_ratio = pd.DataFrame()
#     ratio_fac = 0
#     vals_ = {}
    for sample_ in sample_names_split:
#         pass_ = int(sample_.split("_")[0][1])
#         brep_ = int(sample_.split("_")[1][1])
        A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
        sample_id = sample_names_split.index(sample_)
        tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
        id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                              (df_speciesAbun_next[sample_].values == 1e-8))[0]
        id_keep = list(set(id_species) - set(id_notkeep))
#         tmp = df_speciesAbun_ratio_tmp_[sample_].values
        tmp = np.array(tmp).reshape((len(id_species), 1))
        tmp = tmp[id_keep, :]
        # vals_ = np.ones((len(id_species)))
        vals_ = np.zeros((len(id_species)))
        vals_tmp = np.multiply((np.matmul(A_train_sample, Ri).flatten())[id_keep], \
                               tmp.flatten()**(1 - p_tmp))
        vals_tmp = np.log10((np.matmul(A_train_sample, Ri).flatten())[id_keep]) + \
            (1 - p_tmp) * np.log10(tmp.flatten())
        vals_[id_keep] = vals_tmp.copy()
        # vals_[vals_ == 0] = 1e-6
#         vals_ = df_speciesAbun_ratio_tmp_[sample_].values
#         vals_[sample_] = (np.matmul(A_train_sample, Ri).flatten())**(1 / p_tmp)

        if norm_:
#             mean_ = 10 ** np.mean(np.log10(vals_[id_keep]))
#             ratio_fac = 10 ** (ratio_means_[sample_] - np.mean(np.log10(vals_[id_keep])))
#             vals_[id_keep] = np.power(10, (np.log10(vals_[id_keep]) * ratio_means_[sample_] / \
#                                     np.mean(np.log10(vals_[id_keep]))))
#     #         vals_ += 1 - np.sum(vals_ * df_speciesAbun_prev[sample_].values)
            vals_next = vals_.copy() * df_speciesAbun_prev[sample_].values
            ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_ * ratio_fac
    
            # ratio_fac = ratio_means_[sample_] / np.sum(np.log10(vals_[id_keep]))
            # vals_ = 10**(np.log10(vals_) * ratio_fac)
            # df_growth_ratio[sample_] = vals_
        else:
#             vals_next = vals_ * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) * ratio_means_[sample_] / \
#                                           np.mean(np.log10(vals_)))
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) + ratio_means_[sample_] - \
#                                           np.mean(np.log10(vals_)))
    
#     growth_ratios_new = np.zeros(())
    
    return df_growth_ratio


def compute_inv_growth_ratio_iterate_blind(df_speciesAbun_prev, \
                                           df_speciesAbun_next, p_tmp, Ri, inv_growth_ratios_, \
                                           ratio_means_, df_speciesMetab, norm_=True):
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = 1 / inv_growth_ratios_.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (1 / inv_growth_ratios_.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
    sample_names_split, mat_cons_abun_split_list_tmp = \
        iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                             df_speciesAbun_next.copy(), p_=p_tmp)

    id_species = range(df_speciesAbun_prev.shape[0])
    count_ = 0
    df_inv_growth_ratio = pd.DataFrame()
#     ratio_fac = 0
#     vals_ = {}
    for sample_ in sample_names_split:
#         pass_ = int(sample_.split("_")[0][1])
#         brep_ = int(sample_.split("_")[1][1])
        A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
        sample_id = sample_names_split.index(sample_)
        tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
        id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                              (df_speciesAbun_next[sample_].values == 1e-8))[0]
        id_keep = list(set(id_species) - set(id_notkeep))
#         tmp = df_speciesAbun_ratio_tmp_[sample_].values
        tmp = np.array(tmp).reshape((len(id_species), 1))
        tmp = tmp[id_keep, :]
        vals_ = np.ones((len(id_species)))
        vals_tmp = np.multiply((np.matmul(A_train_sample, Ri).flatten())[id_keep], \
                               tmp.flatten()**(1 - p_tmp))
        vals_[id_keep] = vals_tmp.copy()
        vals_[vals_ == 0] = 1e-6
#         vals_ = df_speciesAbun_ratio_tmp_[sample_].values
#         vals_[sample_] = (np.matmul(A_train_sample, Ri).flatten())**(1 / p_tmp)

        if norm_:
#             mean_ = 10 ** np.mean(np.log10(vals_[id_keep]))
#             ratio_fac = 10 ** (ratio_means_[sample_] - np.mean(np.log10(vals_[id_keep])))
#             vals_[id_keep] = np.power(10, (np.log10(vals_[id_keep]) * ratio_means_[sample_] / \
#                                     np.mean(np.log10(vals_[id_keep]))))
#     #         vals_ += 1 - np.sum(vals_ * df_speciesAbun_prev[sample_].values)
            vals_next = vals_.copy() * df_speciesAbun_prev[sample_].values
            ratio_fac = 1 / np.sum(vals_next)
            df_inv_growth_ratio[sample_] = vals_ * ratio_fac
    
            # ratio_fac = ratio_means_[sample_] / np.sum(np.log10(vals_[id_keep]))
            # vals_ = 10**(np.log10(vals_) * ratio_fac)
            # df_growth_ratio[sample_] = vals_
        else:
#             vals_next = vals_ * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
            df_inv_growth_ratio[sample_] = 1 / vals_
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) * ratio_means_[sample_] / \
#                                           np.mean(np.log10(vals_)))
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) + ratio_means_[sample_] - \
#                                           np.mean(np.log10(vals_)))
    
#     growth_ratios_new = np.zeros(())
    
    return df_inv_growth_ratio


def compute_growth_ratio_iterate_blind_seq(df_speciesAbun_prev, \
                                           df_speciesAbun_next, p_tmp, Ri, \
                                           df_speciesMetab, num_iter=100):
    num_species = df_speciesAbun_prev.shape[0]
    num_metabs = df_speciesMetab.shape[1]
    growth_ratios_tmp = pd.DataFrame()
    for col_ in df_speciesAbun_prev.columns.values:
        growth_ratios_tmp[col_] = np.ones((num_species))
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = growth_ratios_tmp.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (growth_ratios_tmp.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
        
    growth_rate_all = {}
    growth_rate_all[0] = pd.DataFrame()
    for col_ in df_speciesAbun_prev.columns.values:
        growth_rate_all[0][col_] = np.ones((num_species))
        
    for iter_ in range(num_iter):
        id_order_species = \
            np.random.choice(np.arange(num_species), num_species, replace=False)
        for id_species in id_order_species:
            for col_ in range(df_speciesAbun_prev.shape[1]):
                df_speciesAbun_ratio_new.iloc[:, col_] = \
                    growth_ratios_tmp.iloc[:, col_].values
                df_speciesAbun_next.iloc[:, col_] = \
                    (growth_ratios_tmp.iloc[:, col_].values) * \
                    df_speciesAbun_prev.iloc[:, col_].values
            
            sample_names_split, mat_cons_abun_split_list_tmp = \
                iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                                    df_speciesAbun_next.copy(), p_=p_tmp)

            # id_species = range(df_speciesAbun_prev.shape[0])
            count_ = 0
            df_growth_ratio = pd.DataFrame()
        #     ratio_fac = 0
        #     vals_ = {}
            for sample_ in sample_names_split:
        #         pass_ = int(sample_.split("_")[0][1])
        #         brep_ = int(sample_.split("_")[1][1])
                A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
                A_train_sample = A_train_sample.reshape((len([id_species]), num_metabs))
                sample_id = sample_names_split.index(sample_)
                tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
                id_notkeep = \
                    np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                             (df_speciesAbun_next[sample_].values == 1e-8))[0]
                id_keep = list(set([id_species]) - set(id_notkeep))
        #         tmp = df_speciesAbun_ratio_tmp_[sample_].values
                if len(id_keep) > 0:
                    tmp = np.array(tmp).reshape((len([id_species]), 1))
                    vals_ = np.ones((len([id_species])))
                    vals_tmp = \
                        np.multiply((np.matmul(A_train_sample, Ri).flatten()), \
                                    tmp.flatten()**(1 - p_tmp))
                    vals_ = vals_tmp.copy()
                else:
                    vals_ = np.ones((len([id_species])))
                growth_ratios_tmp.iloc[id_species, sample_id] = vals_
        growth_rate_all[iter_] = growth_ratios_tmp
    
    return growth_rate_all

def compute_growth_ratio_iterate(df_speciesAbun_prev, df_speciesAbun_next, p_tmp, Ri, growth_ratios_, \
                                 ratio_means_, df_speciesMetab, norm_=True):
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = growth_ratios_.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (growth_ratios_.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
    sample_names_split, mat_cons_abun_split_list_tmp = \
        iterate_growth_ratio(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                             df_speciesAbun_next.copy(), p_=p_tmp)

    id_species = range(df_speciesAbun_prev.shape[0])
    count_ = 0
    df_growth_ratio = pd.DataFrame()
#     ratio_fac = 0
#     vals_ = {}
    for sample_ in sample_names_split:
        pass_ = int(sample_.split("_")[0][1])
        brep_ = int(sample_.split("_")[1][1])
        A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
        sample_id = sample_names_split.index(sample_)
        tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
        id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                              (df_speciesAbun_next[sample_].values == 1e-8))[0]
        id_keep = list(set(id_species) - set(id_notkeep))
#         tmp = df_speciesAbun_ratio_tmp_[sample_].values
        tmp = np.array(tmp).reshape((len(id_species), 1))
        tmp = tmp[id_keep, :]
        vals_ = np.ones((len(id_species)))
        vals_tmp = np.multiply((np.matmul(A_train_sample, Ri).flatten())[id_keep], \
                               tmp.flatten()**(1 - p_tmp))
        vals_[id_keep] = vals_tmp.copy()
#         vals_ = df_speciesAbun_ratio_tmp_[sample_].values
#         vals_[sample_] = (np.matmul(A_train_sample, Ri).flatten())**(1 / p_tmp)

        if norm_:
#             mean_ = 10 ** np.mean(np.log10(vals_[id_keep]))
#             ratio_fac = 10 ** (ratio_means_[sample_] - np.mean(np.log10(vals_[id_keep])))
#             vals_[id_keep] = np.power(10, (np.log10(vals_[id_keep]) * ratio_means_[sample_] / \
#                                     np.mean(np.log10(vals_[id_keep]))))
#     #         vals_ += 1 - np.sum(vals_ * df_speciesAbun_prev[sample_].values)
#             vals_next = vals_.copy() * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
#             df_growth_ratio[sample_] = vals_ * ratio_fac
    
            ratio_fac = ratio_means_[sample_] / np.mean(np.log10(vals_[id_keep]))
            vals_ = 10**(np.log10(vals_) * ratio_fac)
            df_growth_ratio[sample_] = vals_
        else:
#             vals_next = vals_ * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) * ratio_means_[sample_] / \
#                                           np.mean(np.log10(vals_)))
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) + ratio_means_[sample_] - \
#                                           np.mean(np.log10(vals_)))
    
#     growth_ratios_new = np.zeros(())
    
    return df_growth_ratio

def compute_growth_ratio_iterate_new(df_speciesAbun_prev, df_speciesAbun_next, p_tmp, Ri, growth_ratios_, \
                                     ratio_means_, df_speciesMetab, norm_=True):
    df_speciesAbun_ratio_new = df_speciesAbun_prev.copy()
    for col_ in range(df_speciesAbun_prev.shape[1]):
        df_speciesAbun_ratio_new.iloc[:, col_] = growth_ratios_.iloc[:, col_].values
        df_speciesAbun_next.iloc[:, col_] = (growth_ratios_.iloc[:, col_].values) * \
            df_speciesAbun_prev.iloc[:, col_].values
    sample_names_split, mat_cons_abun_split_list_tmp = \
        iterate_growth_ratio_pvec(df_speciesMetab.copy(), df_speciesAbun_prev.copy(), \
                                  df_speciesAbun_next.copy(), p_vec=p_tmp)

    id_species = range(df_speciesAbun_prev.shape[0])
    count_ = 0
    df_growth_ratio = pd.DataFrame()
#     ratio_fac = 0
#     vals_ = {}
    num_metabs = df_speciesMetab.shape[1]
    for sample_ in sample_names_split:
        pass_ = int(sample_.split("_")[0][1])
        brep_ = int(sample_.split("_")[1][1])
        A_train_sample = mat_cons_abun_split_list_tmp[sample_][id_species, :]
        sample_id = sample_names_split.index(sample_)
        tmp = np.array(df_speciesAbun_ratio_new.iloc[id_species, sample_id])
        id_notkeep = np.where((df_speciesAbun_prev[sample_].values == 1e-8) & \
                              (df_speciesAbun_next[sample_].values == 1e-8))[0]
        id_keep = list(set(id_species) - set(id_notkeep))
#         tmp = df_speciesAbun_ratio_tmp_[sample_].values
        tmp = np.array(tmp).reshape((len(id_species), 1))
        tmp = tmp[id_keep, :]
        vals_ = np.ones((len(id_species)))
        tmp_mat = tmp.flatten()
        tmp_mat = np.tile(tmp_mat, num_metabs).reshape((num_metabs, len(tmp_mat))).T
        p_tmp_mat = \
            np.tile(p_tmp.flatten(), len(tmp)).reshape((len(tmp), len(p_tmp)))
        A_tmp = np.multiply(A_train_sample[id_keep, :], tmp_mat**(1 - p_tmp_mat))
        vals_tmp = (np.matmul(A_tmp, Ri).flatten())
#         vals_tmp = np.multiply((np.matmul(A_train_sample, Ri).flatten())[id_keep], \
#                                tmp.flatten()**(1 - p_tmp))
        vals_[id_keep] = vals_tmp.copy()
#         vals_ = df_speciesAbun_ratio_tmp_[sample_].values
#         vals_[sample_] = (np.matmul(A_train_sample, Ri).flatten())**(1 / p_tmp)

        if norm_:
#             mean_ = 10 ** np.mean(np.log10(vals_[id_keep]))
#             ratio_fac = 10 ** (ratio_means_[sample_] - np.mean(np.log10(vals_[id_keep])))
#             vals_[id_keep] = np.power(10, (np.log10(vals_[id_keep]) * ratio_means_[sample_] / \
#                                     np.mean(np.log10(vals_[id_keep]))))
#     #         vals_ += 1 - np.sum(vals_ * df_speciesAbun_prev[sample_].values)
#             vals_next = vals_.copy() * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
#             df_growth_ratio[sample_] = vals_ * ratio_fac
    
            ratio_fac = ratio_means_[sample_] / np.mean(np.log10(vals_[id_keep]))
            vals_ = 10**(np.log10(vals_) * ratio_fac)
            df_growth_ratio[sample_] = vals_
        else:
#             vals_next = vals_ * df_speciesAbun_prev[sample_].values
#             ratio_fac = 1 / np.sum(vals_next)
            df_growth_ratio[sample_] = vals_
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) * ratio_means_[sample_] / \
#                                           np.mean(np.log10(vals_)))
#         df_growth_ratio[sample_] = 10 ** (np.log10(vals_) + ratio_means_[sample_] - \
#                                           np.mean(np.log10(vals_)))
    
#     growth_ratios_new = np.zeros(())
    
    return df_growth_ratio

def hierarchical_cluster_metabs(df_speciesMetab, n_clusters, metric="euclidean", \
                                method="ward"):
    plt_ = sns.clustermap(df_speciesMetab, cmap="coolwarm", figsize=(10, 10), \
                          metric="euclidean")
    linkage_data = linkage(df_speciesMetab.T, method="ward", metric=metric)
#     dendrogram(linkage_data)
#     plt.show()
    
    h_cluster = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage="ward")
    cluster_labels_new = h_cluster.fit_predict(df_speciesMetab.T)
    
    return cluster_labels_new

def avg_consumption_df(df_speciesMetab, df_speciesMetab_prod, df_metabs_clusters, \
                                             metab_cluster_mean_func="linear"):
        """
        Calculate the average consumption of metabolites by species and their production,
        based on clusters of metabolites.

        Parameters:
        - df_speciesMetab (pd.DataFrame): DataFrame containing species metabolite data.
        - df_speciesMetab_prod (pd.DataFrame): DataFrame containing species metabolite production data.
        - df_metabs_clusters (pd.DataFrame): DataFrame containing metabolite clusters.
        - metab_cluster_mean_func (str, optional): Method to calculate the mean consumption/production
            within each metabolite cluster. Can be "geometric" or "linear". Defaults to "linear".

        Returns:
        - df_speciesMetab_cluster (pd.DataFrame): DataFrame containing the average consumption of
            metabolites by species, grouped by metabolite clusters.
        - df_speciesMetab_prod_cluster (pd.DataFrame): DataFrame containing the average production of
            metabolites by species, grouped by metabolite clusters.
        """
        num_species = df_speciesMetab.shape[0]
        # df_speciesMetab_tmp = df_speciesMetab.copy()
        # df_speciesMetab_tmp[df_speciesMetab_tmp == 0] = 1e-6
        df_speciesMetab_cluster = pd.DataFrame()
        df_speciesMetab_prod_cluster = pd.DataFrame()
        for metab_label in range(df_metabs_clusters.shape[0]):
                id_metabs = df_metabs_clusters.iloc[metab_label, 2]
                consm_vec = np.zeros((num_species))
                consm_vec_prod = np.zeros((num_species))
                
                if metab_cluster_mean_func == "geometric":
                        for id_ in id_metabs:
                                consm_vec += np.log(1 - df_speciesMetab.iloc[:, id_].values)
                                consm_vec_prod += np.log(1 + df_speciesMetab_prod.iloc[:, id_].values)
                        consm_vec /= len(id_metabs)
                        consm_vec = 1 - np.exp(consm_vec)
                        consm_vec_prod /= len(id_metabs)
                        consm_vec_prod = np.exp(consm_vec_prod) - 1
                        df_speciesMetab_cluster[metab_label] = consm_vec.copy()
                        df_speciesMetab_prod_cluster[metab_label] = consm_vec_prod.copy()
                elif metab_cluster_mean_func == "linear":
                        for id_ in id_metabs:
                                consm_vec += df_speciesMetab.copy().iloc[:, id_].values
                                consm_vec_prod += df_speciesMetab_prod.copy().iloc[:, id_].values
                        consm_vec /= len(id_metabs)
                        consm_vec_prod /= len(id_metabs)
                        df_speciesMetab_cluster[metab_label] = consm_vec.copy()
                        df_speciesMetab_prod_cluster[metab_label] = consm_vec_prod.copy()
        df_speciesMetab_cluster.index = df_speciesMetab.index.values
        df_speciesMetab_prod_cluster.index = df_speciesMetab.index.values
        return df_speciesMetab_cluster, df_speciesMetab_prod_cluster

def get_metabs_clusters(df_speciesMetab, df_speciesMetab_prod, bin_thresh=0.3, \
                        species_num_thresh=5, \
                        n_clusters_hclust=10, distance_metric="euclidean", \
                        method_cluster="ward", normalize_=False):
    """
    Perform clustering analysis on species-metabolite consumption matrix, c_{\\alpha, i}.

    Args:
        df_speciesMetab (pd.DataFrame): DataFrame containing 
                                        species-metabolite consumption matrix, c_{\\alpha, i}.
        df_speciesMetab_prod (pd.DataFrame): DataFrame containing species-metabolite 
                                             production matrix, p_{\\gamma, i}.
        bin_thresh (float, optional): Threshold for binarizing the species-metabolite consumption matrix. 
                                      Defaults to 0.3.
        species_num_thresh (int, optional): Threshold for the minimum number of species required 
                                            to treat a metabolite as a non-singleton cluster. Defaults to 5.
        n_clusters_hclust (int, optional): Number of non-singleton clusters to be formed using 
                                           hierarchical clustering. Defaults to 10.
        distance_metric (str, optional): Distance metric to be used for clustering. Defaults to "euclidean".
        method_cluster (str, optional): Linkage method to be used for clustering. Defaults to "ward".
        normalize_ (bool, optional): Flag indicating whether to normalize the data. Defaults to False.

    Returns:
        tuple: A tuple containing the following:
            - df_metabs_clusters (pd.DataFrame): DataFrame containing information about the clusters.
            - df_speciesMetab_new (pd.DataFrame): DataFrame containing species-metabolite consumption matrix 
                                                  after clustering.
            - df_speciesMetab_prod_new (pd.DataFrame): DataFrame containing species-metabolite production 
                                                       matrix after clustering.
    """
    # binarize the species-metabolite consumption matrix
    df_tmp = df_speciesMetab.copy()
    df_tmp[df_tmp < bin_thresh] = 0
    df_tmp[df_tmp >= bin_thresh] = 1

    # get the number of species consuming each metabolite
    sum_ = np.sum(np.array(df_tmp), axis=0)

    # get the indices of metabolites to be part of non-singleton clusters
    id_metabs_clust = np.where(sum_ >= species_num_thresh)[0]

    # get the indices of metabolites to be removed
    # id_metabs_rm = np.where(sum_ == 0)[0]

    # get the indices of metabolites to be part of singleton clusters
    id_metabs_single_clust = np.where((sum_ < species_num_thresh) & \
                                      (sum_ > 0))[0]
    
    # perform hierarchical clustering
    h_cluster = AgglomerativeClustering(n_clusters=n_clusters_hclust, \
                                        metric=distance_metric, \
                                        linkage=method_cluster)
    df_tmp_clust = df_speciesMetab.copy()
    df_tmp_clust = df_tmp_clust.iloc[:, id_metabs_clust]
    if not normalize_:
        df_tmp_ = df_tmp_clust.copy()
        df_tmp_ = df_tmp_.transpose()
    else:
        df_tmp_un = df_tmp_clust.copy()
        df_tmp_un = df_tmp_un.transpose()
        df_tmp_ = (df_tmp_un - df_tmp_un.mean())/df_tmp_un.std()
    
    # get the cluster labels
    cluster_labels = h_cluster.fit_predict(df_tmp_)
    clust_labels_un, clust_counts = np.unique(cluster_labels, return_counts=True)

    cluster_labels_new = np.hstack([cluster_labels, \
                                    np.arange(len(clust_labels_un), \
                                              len(clust_labels_un) + \
                                                len(id_metabs_single_clust))])
    
    # get the new indices for the species-metabolite consumption matrix
    id_new = np.hstack([id_metabs_clust, id_metabs_single_clust])
    
    # redorder the species-metabolite consumption and production matrices based on the new indices
    # with all metabolites in the non-singleton clusters first, followed by the singleton clusters
    df_speciesMetab_new = df_speciesMetab.copy()
    df_speciesMetab_new = df_speciesMetab_new.iloc[:, id_new]
    df_speciesMetab_prod_new = df_speciesMetab_prod.copy()
    df_speciesMetab_prod_new = df_speciesMetab_prod_new.iloc[:, id_new]
    metab_names = np.hstack([df_tmp_clust.columns.values, \
                             df_speciesMetab.columns.values[id_metabs_single_clust]])
    
    clust_labels_un, clust_counts = np.unique(cluster_labels_new, return_counts=True)

    # dataframe for clusters with metabolites in each cluster
    df_metabs_clusters = pd.DataFrame()
    df_metabs_clusters["cluster_labels"] = clust_labels_un
    df_metabs_clusters["cluster_counts"] = clust_counts

    

    cluster_metab_IDs = []
    cluser_metab_names = []
    for label_ in clust_labels_un:
        id_label = np.where(cluster_labels_new == label_)[0]
        cluster_metab_IDs.append(list(id_label))
        cluser_metab_names.append(list(metab_names[id_label]))  
        
    df_metabs_clusters["cluster_metab_IDs"] = cluster_metab_IDs
    df_metabs_clusters["cluser_metab_names"] = cluser_metab_names

    return df_metabs_clusters, df_speciesMetab_new, df_speciesMetab_prod_new

def fit_ss_Ri(df_speciesMetab_cluster, df_speciesAbun_mdl, \
              df_speciesAbun_prev_mdl, df_speciesAbun_next_mdl, \
              df_speciesAbun_ratio_mdl, \
              file_save, num_passages=6, pass_rm=[0, 1, 2]):
    num_species = df_speciesMetab_cluster.shape[0]
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    num_metabs = df_speciesMetab_tmp.shape[1]

    # data for steady state fit
    pass_rm = [0, 1, 2]
    pass_keep = remove_passages(pass_rm, num_passages=num_passages)
    df_speciesAbun_tmp = df_speciesAbun_mdl.copy().iloc[:, pass_keep]

    Ri_noMicrocosm_steadyState_fit_all = np.zeros((num_species, num_metabs))
    Ri_noMicrocosm_steadyState_fit_avg = np.zeros((num_metabs))
    Ri_noMicrocosm_steadyState_fit_joint = \
        compute_Ri_ss(df_speciesMetab_tmp.copy(), \
                        df_speciesAbun_tmp.copy(), \
                        df_speciesAbun_tmp.copy(), \
                        0, \
                        range(num_species))

    count_species = 0
    for species_ in range(num_species):
        id_species = list(set(range(num_species)) - set([species_]))
        Ri_noMicrocosm_steadyState_fit_all[species_, :] = \
            compute_Ri_ss(df_speciesMetab_tmp.copy(), df_speciesAbun_tmp, \
                        df_speciesAbun_tmp, \
                        0, id_species)
        if np.sum(Ri_noMicrocosm_steadyState_fit_all[species_, :]) <= 1.5:
            Ri_noMicrocosm_steadyState_fit_avg += \
                Ri_noMicrocosm_steadyState_fit_all[species_, :]
            count_species += 1


    Ri_noMicrocosm_steadyState_fit_avg /= count_species


    save_obj = {"Ri_noMicrocosm_steadyState_fit_all" : Ri_noMicrocosm_steadyState_fit_all, \
                "Ri_noMicrocosm_steadyState_fit_avg" : Ri_noMicrocosm_steadyState_fit_avg, \
                "Ri_noMicrocosm_steadyState_fit_joint" : Ri_noMicrocosm_steadyState_fit_joint}
    with open(file_save, "wb") as file_:
        pickle.dump(save_obj, file_) 
    
    return save_obj

def match_lhs_rhs_fit(Ri_, df_speciesMetab, \
                      df_speciesAbun_prev, df_speciesAbun_next, \
                      df_speciesAbun_ratio, save_file, \
                      num_passages=6, pass_rm=[0, 1, 2], fit_type="steady_state", p_tmp=0):
    num_species = df_speciesMetab.shape[0]
    pass_keep = remove_passages(pass_rm, num_passages=num_passages)
    # df_speciesAbun_tmp = df_speciesAbun.copy().iloc[:, pass_keep]
    df_speciesAbun_prev_tmp = df_speciesAbun_prev.copy().iloc[:, pass_keep]
    df_speciesAbun_next_tmp = df_speciesAbun_next.copy().iloc[:, pass_keep]
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio.copy().iloc[:, pass_keep]
    if fit_type == "steady_state":
        goodness_func = compute_Ri_ss_goodness_of_fit
    elif fit_type == "dynamic":
        goodness_func = compute_Ri_dynamic_goodness_of_fit
    
    lhs_, rhs_, _, _ = \
        goodness_func(df_speciesMetab, \
                      df_speciesAbun_prev_tmp, df_speciesAbun_next_tmp, \
                      df_speciesAbun_ratio_tmp, \
                      range(num_species), \
                      Ri_, p_tmp=p_tmp)
    # lhs_, rhs_, _, _ = \
    #     compute_Ri_ss_goodness_of_fit(df_speciesMetab, \
    #                                     df_speciesAbun_tmp, df_speciesAbun_tmp, \
    #                                     range(num_species), \
    #                                     Ri_)
    
    
    fig, axes = plt.subplots(1, 1, figsize=(14, 16), sharey="row", sharex="col")
    # fig.suptitle(f'RHS vs LHS for Ri fitted steadyState for steadyStates', \
    #             fontsize=20)
    fig.supylabel('RHS (log scale)', fontsize=30)
    fig.supxlabel('LHS (log scale)', fontsize=30)

    x = np.log10(lhs_)
    y = np.log10(rhs_)

    plt_ = sns.scatterplot(x=x, \
                           y=y, ax=axes, s=30)

    plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                    ax=axes)
    if fit_type == "steady_state":
        plt_.plot([-8, 0], [-8, 0], c="green", linewidth=3)
    elif fit_type == "dynamic":
        plt_.plot([-4, 1.7], [-4, 1.7], c="green", linewidth=3)
    #             plt_ = sns.lineplot(x = x_order[id_not_nan], y=rolling_avg[id_not_nan], c="black")

    plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=30)
    plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=30)
    corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
    corr_val_pe_log = scipy.stats.pearsonr(x, y)
    corr_val_sp = scipy.stats.spearmanr(x, y)

    model = sm.OLS(rhs_, lhs_).fit()
    slope = model.params[0]
    slope_pval = model.pvalues[0]
    rms_ = np.sqrt(np.mean(np.power(np.log10(lhs_) - np.log10(rhs_), 2)))


    # plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope)], c="green")

    # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
    #                 '{:.3e}'.format(corr_val_pe[1]) + \
    #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
    #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
    #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
    #                 '{:.3e}'.format(corr_val_sp[1]) + \
    #         f'\n slope = {np.round(slope, 3)}, pvalue = ' + \
    #                 '{:.3e}'.format(slope_pval) + f', RMSE = {np.round(rms_, 3)}'
    # if fit_type == "steady_state":
    #     title_ = f'RHS vs LHS for fitted Ri, ' + \
    #          r'$f = {f_val}$'.format(f_val=fval) + \
    #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
    #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
    #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
    #                 '{:.3e}'.format(corr_val_sp[1]) + \
    #         f'\n, RMSE = {np.round(rms_, 3)}'
    # elif fit_type == "dynamic":
    fval = 1 - p_tmp
    title_ = f'RHS vs LHS for fitted Ri with ' + \
            r'$f = {f_val}$'.format(f_val=fval) + \
        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                '{:.3e}'.format(corr_val_pe_log[1]) + \
        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                '{:.3e}'.format(corr_val_sp[1]) + \
        f'\n RMSE = {np.round(rms_, 3)}'

    axes.set_title(title_, size=30)

    fig.figure.savefig(save_file, dpi=300, transparent=False, facecolor="white")
    plt.close(fig.figure)

def blindly_pred_abun_growth_no_f(p_vec_new, df_speciesMetab_cluster, \
                             df_speciesAbun_inoc, df_speciesAbun_mdl, \
                             df_speciesAbun_prev_mdl, \
                             df_speciesAbun_ratio_mdl, \
                             Ri_, dir_save_abun_obj, \
                             dir_save_abun, \
                             dir_save_growth, \
                             num_passages=6, num_iter=100, \
                             thresh_zero=1e-10, Ri_ss=True, plot_=True, \
                             save_data_obj=True, \
                             return_sensitivity_ana=False, \
                             get_prod=False, \
                             B_alone=None, \
                             df_speciesMetab_prod=None, \
                             prod_use_prev=True, \
                             num_passages_run=6, use_dilution=False, \
                             dilution_factor=15000, \
                             id_species_update=None, \
                             use_avg_for_prod=True, \
                             Ri_0=None):
    if Ri_0 is None:
        Ri_0 = Ri_.copy()
    num_species = df_speciesMetab_cluster.shape[0]
    # if id_species_update is None:
    #     id_species_update = np.arange(num_species)
    if id_species_update is not None:
        id_species_noupdate = \
            np.array(list(set(range(num_species)) - \
                 set(list(id_species_update))), dtype='long')
    # simulate inoculum abundances and initial growth ratios
    sample_names = df_speciesAbun_mdl.columns.values
    n_breps = 3
    samples_first = sample_names[0 + np.arange(n_breps) * num_passages]
    # ratio_init = np.zeros((num_species))
    # num_rand = 100
    # for rand_ in range(num_rand):
    #     ratio_init_rand = np.zeros((num_species, n_breps))
    #     for rep_ in range(n_breps):
    #         vals_ = np.random.exponential(1 / num_species, num_species)
    #         vals_ /= np.sum(vals_)
    #         ratio_init_rand[:, rep_] = df_speciesAbun_mdl[samples_first[rep_]].values / \
    #             vals_
    #     ratio_init += np.mean(np.log10(ratio_init_rand), axis=1)
    # ratio_init /= num_rand
    # ratio_init = 10**ratio_init
    ratio_init = np.ones(num_species)
    # print(ratio_init)


    df_corr_slope = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})

    df_corr_slope_growth = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})
    
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    if return_sensitivity_ana:
        save_obj_return = {}
        RMSE_obj = {}
    for count_p, p_tmp in enumerate(p_vec_new):
        for pass_ in range(num_passages_run):
            if pass_ == 0:
                df_speciesAbun_prev_tmp_ = \
                    df_speciesAbun_inoc.copy().iloc[:, :]
            else:
                growth_ratio_prev_ = growth_rate_all[num_iter - 1].copy()
                df_tmp = growth_rate_all[num_iter - 1].copy()
                for col_ in df_tmp.columns.values:
                    if id_species_update is not None:
                        df_tmp[col_][id_species_noupdate] = \
                            1
                    df_speciesAbun_prev_tmp_[col_] = \
                        df_tmp[col_].values * df_speciesAbun_prev_tmp_[col_].values

                    if id_species_update is None:
                        df_speciesAbun_prev_tmp_[col_] /= \
                            np.sum(df_speciesAbun_prev_tmp_[col_].values)
                    else:
                        df_speciesAbun_prev_tmp_[col_][id_species_update] /= \
                            np.sum(df_speciesAbun_prev_tmp_[col_].values[id_species_update])
            df_speciesAbun_next_tmp_ = df_speciesAbun_prev_tmp_.copy()


            growth_rate_all = {}
            growth_rate_all[0] = pd.DataFrame()
            growth_rate_tmp = pd.DataFrame()
            if Ri_ss:
                Ri_avg = Ri_.copy()
            else:
                if pass_ != 0:
                    Ri_avg = Ri_[count_p].copy()
                else:
                    Ri_avg = Ri_0[count_p].copy()
            
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                # growth_rate_all[0][col_] = np.ones((num_species))
                growth_rate_all[0][col_] = ratio_init
                growth_rate_tmp[col_] = np.ones((num_species))



            for iter_ in range(num_iter):
                if iter_ == 0:
                    iter_id = iter_
                else:
                    iter_id = iter_ - 1
                
                if (iter_ == 0) & (pass_ != 0):
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind_no_f(df_speciesAbun_prev_tmp_.copy(), \
                                                        df_speciesAbun_next_tmp_.copy(), \
                                                        p_tmp, Ri_avg.copy(), \
                                                        # growth_ratio_prev_.copy(), \
                                                        growth_rate_all[iter_id].copy(), \
                                                        None, df_speciesMetab_tmp,
                                                        norm_=False, get_prod=get_prod, \
                                                        B_alone=B_alone, \
                                                        df_speciesMetab_prod=df_speciesMetab_prod, \
                                                        prod_use_prev=prod_use_prev, \
                                                        use_dilution=use_dilution, \
                                                        dilution_factor=dilution_factor, \
                                                        use_avg_for_prod=use_avg_for_prod)
                else:
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind_no_f(df_speciesAbun_prev_tmp_.copy(), \
                                                           df_speciesAbun_next_tmp_.copy(), \
                                                           p_tmp, Ri_avg.copy(), \
                                                           growth_rate_all[iter_id].copy(), \
                                                           None, df_speciesMetab_tmp,
                                                           norm_=False, get_prod=get_prod, \
                                                           B_alone=B_alone, \
                                                           df_speciesMetab_prod=df_speciesMetab_prod, \
                                                           prod_use_prev=prod_use_prev, \
                                                           use_dilution=use_dilution, \
                                                           dilution_factor=dilution_factor, \
                                                           use_avg_for_prod=use_avg_for_prod)

                growth_rate_all[iter_] = df_growth_rate.copy()
            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun_obj, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             'predicted_abundance', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_Ri_fit' + \
                                                     f'_with_p{p_tmp}.pickle'))
            
            save_obj = {'growth_rate_all' : growth_rate_all, \
                        'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            # if return_sensitivity_ana:
            #     save_obj_return[pass_] = \
            #         {'growth_rate_all' : growth_rate_all[num_iter - 1], \
            #          'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            if save_data_obj:
                with open(file_save, "wb") as file_:
                    pickle.dump(save_obj, file_)  
            
            
            df_tmp = pd.DataFrame()
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                tmp_ = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values
                # print(np.sum((growth_rate_all[num_iter - 1].copy()[col_].values * \
                #               df_speciesAbun_prev_tmp_[col_].values) / \
                #         np.sum(tmp_)))
                    
                if id_species_update is None:
                    growth_rate_all[num_iter - 1][col_] = \
                        growth_rate_all[num_iter - 1].copy()[col_].values / \
                        np.sum(tmp_)
                    # df_tmp[col_] /= np.sum(df_tmp[col_].values)
                else:
                    growth_rate_all[num_iter - 1][col_] = \
                        growth_rate_all[num_iter - 1].copy()[col_].values / \
                        np.sum(tmp_[id_species_update])
                    growth_rate_all[num_iter - 1][col_][id_species_noupdate] = 1
                    # df_tmp[col_][id_species_update] /= \
                    #     np.sum(df_tmp[col_].values[id_species_update])
                df_tmp[col_] = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values
                # print(np.sum(df_tmp[col_].values))

            b_ = range(3)
            x = \
                np.array(df_speciesAbun_mdl.copy().iloc[:, [pass_, \
                                                            pass_ + num_passages, \
                                                            pass_ + 2 * num_passages]])
            x[x == 0] = thresh_zero
            # x = 10**(np.mean(np.log10(x), axis=1)).flatten()
            x = x.flatten()
            # y = np.array(df_tmp.copy())[:, :].flatten()
            y = np.array(df_tmp.copy())[:, :]
            y = np.hstack([y, y, y]).flatten()
            y[y == 0] = thresh_zero
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                save_obj_return[pass_] = \
                    {'growth_ratio_all' : growth_rate_all[num_iter - 1], \
                     'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_, \
                     'df_speciesAbun_next_pred' : y, \
                     'df_speciesAbun_next_obs' : x, \
                     'species_names' : species_names}
            # print(x.shape)
            
            id_ = np.where((x > 0) & (y > 0))[0]
            # id_ord = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # zero_thresh = -10
            y[y <= np.log10(thresh_zero)] = np.log10(thresh_zero)

            # if pass_ > 0:
            #     pass_tmp = pass_ - 1
            #     df_speciesAbun_ratio_tmp_1 = \
            #         df_speciesAbun_ratio_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
            #                                                  pass_tmp + 10]]
            #     x_r = \
            #         np.array(df_speciesAbun_ratio_tmp_1)
            #     x_new = -1 * np.ones(x_r.shape[0])
            #     for row_ in range(x_r.shape[0]):
            #         id_tmp = np.where(x_r[row_, :] > 0)[0]
            #         if len(id_tmp) > 0:
            #             x_new[row_] = 10**np.mean(np.log10(x_r[row_, id_tmp]))
            #     # id_tmp = np.where(x_new > 0)[0]
            #     # x_r = x_new[id_tmp]
            #     x_r = x_new
            #     x_r[x_r < 0] = 1 
            # else:
            #     x_r = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]])
            #     x_r = 10**np.mean(np.log10(x_r), axis=1).flatten()
            #     x_r /= np.array(df_speciesAbun_inoc).flatten()

            # # id_ = np.where((x > 0) & (y > 0))[0]
            # x_r = x_r[id_ord]

            # x_r[x_r > 1] = 3
            # x_r[x_r == 1] = 2
            # x_r[x_r < 1] = 1
            # growth_ord_tmp = [100*n**2 for n in x_r]

            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_] = {}
                RMSE_obj[pass_]["abundance"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted abundance (log scale)', fontsize=30)
                fig.supxlabel('observed abundance (log scale)', fontsize=30)
                # plt_ = sns.scatterplot(x=x, \
                #                        y=y, s=growth_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(x=x, \
                                    y=y, s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-8, 0], [-8, 0], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                # abs_mean_error = np.median(np.abs(y - x))
                std_error = np.sqrt(np.std(np.power(y - x, 2)))

                id_notzero_zero = np.where((x > -8) & (y <= -8))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                df_tmp = \
                    pd.DataFrame(data={"passage" : [pass_] * 8, 
                                    "p" : [p_tmp] * 8, 
                                    "metric_type" : ["corr_pearson_log", 
                                                        "corr_spearman", "slope", "slope_log", \
                                                        "corr_pearson_linear", \
                                                        "RMSE", \
                                                        "RSSE", "FNR"], 
                                    "metric" : [corr_val_pe_log[0], 
                                                corr_val_sp[0], slope, slope_log, \
                                                corr_val_pe[0], abs_mean_error, \
                                                std_error, \
                                                frac_zero],
                                    "pval" : [corr_val_pe_log[1], 
                                                corr_val_sp[1], slope_pval, slope_log_pval, \
                                                corr_val_pe[1], 0, 0, 0]})
                df_corr_slope = pd.concat([df_corr_slope, df_tmp], ignore_index=True)

                # plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
                plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="green", \
                        linewidth=3)

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed abundance at passage {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)

                save_dir = \
                    os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
            
            # Plot growth ratios
            if pass_ > 0:
                pass_tmp = pass_ - 1
                df_speciesAbun_ratio_tmp_1 = \
                    df_speciesAbun_ratio_mdl.iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]].copy()
                abun_prev = df_speciesAbun_prev_mdl.iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]].copy()
                abun_prev = np.array(abun_prev)
                abun_prev[abun_prev == 0] = thresh_zero
                abun_prev = 10**np.mean(np.log10(abun_prev), axis=1)
                x = \
                    np.array(df_speciesAbun_ratio_tmp_1)
                x_new = -1 * np.ones(x.shape[0])
                for row_ in range(x.shape[0]):
                    id_tmp = np.where(x[row_, :] > 0)[0]
                    if len(id_tmp) > 0:
                        x_new[row_] = 10**np.mean(np.log10(x[row_, id_tmp]))
                id_tmp = np.where(x_new > 0)[0]
                x = x_new[id_tmp]
                abun_prev = abun_prev.flatten()[id_tmp]
    #                 x = 10**(np.mean(np.log10(x), axis=1)).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                y = y[id_tmp]
            else:
                x = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]].copy())
                x[x == 0] = thresh_zero
                x = 10**np.mean(np.log10(x), axis=1).flatten()
                x_inoc_tmp = np.array(df_speciesAbun_inoc.copy()).flatten()
                x_inoc_tmp[x_inoc_tmp == 0] = thresh_zero
                x /= x_inoc_tmp
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                abun_prev = x_inoc_tmp

            y[y == 0] = 1e-8
            x[x == 0] = 1e-8
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                if pass_ > 0:
                    save_obj_return[pass_]['growth_ratio_obs'] = \
                        np.array(df_speciesAbun_ratio_tmp_1.copy())
                else:
                    save_obj_return[pass_]['growth_ratio_obs'] = np.hstack([x, x, x])
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # y=np.random.permutation(x)
            y[y <= -4] = -4
            x[x <= -4] = -4
            df_plt = pd.DataFrame(data={"x" : x, "y" : y, \
                                        "abun_prev" : abun_prev[id_]})
            prev_ord = np.argsort(abun_prev)
            prev_ord_tmp = [0.2*n**2 for n in prev_ord]
            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_]["growth_ratio"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted growth ratio (log scale)', fontsize=25)
                fig.supxlabel('observed growth ratio (log scale)', fontsize=25)
                # plt_ = sns.scatterplot(data=df_plt, x="x", \
                #                        y="y", s=prev_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(data=df_plt, x="x", \
                                    y="y", s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-4, 2], [-4, 2], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                std_error = np.sqrt(np.std(np.power(y - x, 2)))
                id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                # df_tmp = \
                #     pd.DataFrame(data={"p" : [p_tmp] * 8, 
                #                     "metric_type" : ["corr_pearson_log", 
                #                                         "corr_spearman", "slope", "slope_log", \
                #                                         "corr_pearson_linear", \
                #                                         "RMSE", \
                #                                         "std_error", "FNR"], 
                #                     "metric" : [corr_val_pe_log[0], 
                #                                 corr_val_sp[0], slope, slope_log, \
                #                                 corr_val_pe[0], abs_mean_error, \
                #                                 std_error, \
                #                                 frac_zero],
                #                     "pval" : [corr_val_pe_log[1], 
                #                                 corr_val_sp[1], slope_pval, slope_log_pval, \
                #                                 corr_val_pe[1], 0, 0, 0]})
                # df_corr_slope_growth = \
                #     pd.concat([df_corr_slope_growth, df_tmp], ignore_index=True)


                # plt_.plot([-4, 2], [np.log10(slope * (1e-4)), np.log10(slope * 1e2)], c="green")
                plt_.plot([-4, 2], [(slope_log * (-4)), (slope_log * 2)], c="green", \
                        linewidth=3)
                plt_.axhline(y=0, linestyle="dashed", c="black")
                plt_.axvline(x=0, linestyle="dashed", c="black")

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed growth ratio from {pass_} to {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)


                save_dir = \
                    os.path.abspath(os.path.join(dir_save_growth, \
                                                'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
                                                f'passage_{pass_}->{pass_ + 1}'))
                
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_growth_vs_' + \
                                                        f'observed_growth' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
    if plot_:
        for pass_ in range(num_passages):
            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.suptitle(f'corr, slope for predicted vs observed abundance \n' + \
                        f' passage {pass_ + 1}', \
                        fontsize=30)
            fig.supxlabel('p', fontsize=30)
            fig.supylabel('correlation or slope', fontsize=30)
            df_corr_slope_tmp = df_corr_slope.copy()
            df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
            
            df_corr_slope_tmp = \
                df_corr_slope_tmp[(df_corr_slope_tmp['metric_type'] != "corr_pearson_linear") & \
                                (df_corr_slope_tmp['metric_type'] != "slope")]
            plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_.set_xscale("log", base=10)

            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_stats.png'))
            fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)

    del df_corr_slope

    if return_sensitivity_ana:
        return save_obj_return, RMSE_obj
    # for pass_ in range(num_passages):
    #     fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
    #     fig.suptitle(f'corr, slope for predicted vs observed growth ratio \n' + \
    #                 f' passage {pass_ + 1}', \
    #                 fontsize=30)
    #     fig.supxlabel('p', fontsize=30)
    #     fig.supylabel('correlation or slope', fontsize=30)
    #     df_corr_slope_tmp = df_corr_slope_growth.copy()
    #     df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
    #     plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_.set_xscale("log", base=10)

    #     save_dir = \
    #         os.path.abspath(os.path.join(dir_save_growth, \
    #                                         'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
    #                                         f'passage_{pass_}->{pass_ + 1}'))
        
    #     if not os.path.exists(save_dir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_dir)
        
    #     file_save = os.path.abspath(os.path.join(save_dir, 
    #                                                 f'predicted_growth_vs_' + \
    #                                                 f'observed_growth' + \
    #                                                 f'_stats.png'))
    #     fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
    #     plt.close(fig.figure)

def blindly_pred_abun_growth(p_vec_new, df_speciesMetab_cluster, \
                             df_speciesAbun_inoc, df_speciesAbun_mdl, \
                             df_speciesAbun_prev_mdl, \
                             df_speciesAbun_ratio_mdl, \
                             Ri_, dir_save_abun_obj, \
                             dir_save_abun, \
                             dir_save_growth, \
                             num_passages=6, num_iter=100, \
                             thresh_zero=1e-10, Ri_ss=True, plot_=True, \
                             save_data_obj=True, \
                             return_sensitivity_ana=False, \
                             get_prod=False, \
                             B_alone=None, \
                             df_speciesMetab_prod=None, \
                             prod_use_prev=True, \
                             num_passages_run=6, use_dilution=False, \
                             dilution_factor=15000, \
                             id_species_update=None, \
                             use_avg_for_prod=True, \
                             Ri_0=None):
    if Ri_0 is None:
        Ri_0 = Ri_.copy()
    num_species = df_speciesMetab_cluster.shape[0]
    # if id_species_update is None:
    #     id_species_update = np.arange(num_species)
    if id_species_update is not None:
        id_species_noupdate = \
            np.array(list(set(range(num_species)) - \
                 set(list(id_species_update))), dtype='long')
    # simulate inoculum abundances and initial growth ratios
    sample_names = df_speciesAbun_mdl.columns.values
    n_breps = 3
    samples_first = sample_names[0 + np.arange(n_breps) * num_passages]
    # ratio_init = np.zeros((num_species))
    # num_rand = 100
    # for rand_ in range(num_rand):
    #     ratio_init_rand = np.zeros((num_species, n_breps))
    #     for rep_ in range(n_breps):
    #         vals_ = np.random.exponential(1 / num_species, num_species)
    #         vals_ /= np.sum(vals_)
    #         ratio_init_rand[:, rep_] = df_speciesAbun_mdl[samples_first[rep_]].values / \
    #             vals_
    #     ratio_init += np.mean(np.log10(ratio_init_rand), axis=1)
    # ratio_init /= num_rand
    # ratio_init = 10**ratio_init
    ratio_init = np.ones(num_species)
    # print(ratio_init)


    df_corr_slope = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})

    df_corr_slope_growth = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})
    
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    if return_sensitivity_ana:
        save_obj_return = {}
        RMSE_obj = {}
    for count_p, p_tmp in enumerate(p_vec_new):
        for pass_ in range(num_passages_run):
            if pass_ == 0:
                df_speciesAbun_prev_tmp_ = \
                    df_speciesAbun_inoc.copy().iloc[:, :]
            else:
                growth_ratio_prev_ = growth_rate_all[num_iter - 1].copy()
                df_tmp = growth_rate_all[num_iter - 1].copy()
                for col_ in df_tmp.columns.values:
                    if id_species_update is not None:
                        df_tmp[col_][id_species_noupdate] = \
                            1
                    df_speciesAbun_prev_tmp_[col_] = \
                        df_tmp[col_].values * df_speciesAbun_prev_tmp_[col_].values
                    df_speciesAbun_prev_tmp_[col_][\
                        df_speciesAbun_prev_tmp_[col_].values < thresh_zero] = \
                        thresh_zero

                    if id_species_update is None:
                        df_speciesAbun_prev_tmp_[col_] /= \
                            np.sum(df_speciesAbun_prev_tmp_[col_].values)
                    else:
                        df_speciesAbun_prev_tmp_[col_][id_species_update] /= \
                            np.sum(df_speciesAbun_prev_tmp_[col_].values[id_species_update])
            df_speciesAbun_next_tmp_ = df_speciesAbun_prev_tmp_.copy()


            growth_rate_all = {}
            growth_rate_all[0] = pd.DataFrame()
            growth_rate_tmp = pd.DataFrame()
            if Ri_ss:
                Ri_avg = Ri_.copy()
            else:
                if pass_ != 0:
                    Ri_avg = Ri_[count_p].copy()
                else:
                    Ri_avg = Ri_0[count_p].copy()
            
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                # growth_rate_all[0][col_] = np.ones((num_species))
                growth_rate_all[0][col_] = ratio_init
                growth_rate_tmp[col_] = np.ones((num_species))



            for iter_ in range(num_iter):
                if iter_ == 0:
                    iter_id = iter_
                else:
                    iter_id = iter_ - 1
                
                if (iter_ == 0) & (pass_ != 0):
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                        df_speciesAbun_next_tmp_.copy(), \
                                                        p_tmp, Ri_avg.copy(), \
                                                        # growth_ratio_prev_.copy(), \
                                                        growth_rate_all[iter_id].copy(), \
                                                        None, df_speciesMetab_tmp,
                                                        norm_=False, get_prod=get_prod, \
                                                        B_alone=B_alone, \
                                                        df_speciesMetab_prod=df_speciesMetab_prod, \
                                                        prod_use_prev=prod_use_prev, \
                                                        use_dilution=use_dilution, \
                                                        dilution_factor=dilution_factor, \
                                                        use_avg_for_prod=use_avg_for_prod)
                else:
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                           df_speciesAbun_next_tmp_.copy(), \
                                                           p_tmp, Ri_avg.copy(), \
                                                           growth_rate_all[iter_id].copy(), \
                                                           None, df_speciesMetab_tmp,
                                                           norm_=False, get_prod=get_prod, \
                                                           B_alone=B_alone, \
                                                           df_speciesMetab_prod=df_speciesMetab_prod, \
                                                           prod_use_prev=prod_use_prev, \
                                                           use_dilution=use_dilution, \
                                                           dilution_factor=dilution_factor, \
                                                           use_avg_for_prod=use_avg_for_prod)

                growth_rate_all[iter_] = df_growth_rate.copy()
            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun_obj, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             'predicted_abundance', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_Ri_fit' + \
                                                     f'_with_p{p_tmp}.pickle'))
            
            save_obj = {'growth_rate_all' : growth_rate_all, \
                        'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            # if return_sensitivity_ana:
            #     save_obj_return[pass_] = \
            #         {'growth_rate_all' : growth_rate_all[num_iter - 1], \
            #          'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            if save_data_obj:
                with open(file_save, "wb") as file_:
                    pickle.dump(save_obj, file_)  
            
            
            df_tmp = pd.DataFrame()
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                tmp_ = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values
                # print(np.sum((growth_rate_all[num_iter - 1].copy()[col_].values * \
                #               df_speciesAbun_prev_tmp_[col_].values) / \
                #         np.sum(tmp_)))
                    
                if id_species_update is None:
                    growth_rate_all[num_iter - 1][col_] = \
                        growth_rate_all[num_iter - 1].copy()[col_].values / \
                        np.sum(tmp_)
                    # df_tmp[col_] /= np.sum(df_tmp[col_].values)
                else:
                    growth_rate_all[num_iter - 1][col_] = \
                        growth_rate_all[num_iter - 1].copy()[col_].values / \
                        np.sum(tmp_[id_species_update])
                    growth_rate_all[num_iter - 1][col_][id_species_noupdate] = 1
                    # df_tmp[col_][id_species_update] /= \
                    #     np.sum(df_tmp[col_].values[id_species_update])
                df_tmp[col_] = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values
                # print(np.sum(df_tmp[col_].values))

            b_ = range(3)
            x = \
                np.array(df_speciesAbun_mdl.copy().iloc[:, [pass_, \
                                                            pass_ + num_passages, \
                                                            pass_ + 2 * num_passages]])
            x[x <= thresh_zero] = thresh_zero
            # x = 10**(np.mean(np.log10(x), axis=1)).flatten()
            x = x.flatten()
            # y = np.array(df_tmp.copy())[:, :].flatten()
            y = np.array(df_tmp.copy())[:, :]
            y = np.hstack([y, y, y]).flatten()
            y[y <= thresh_zero] = thresh_zero
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                save_obj_return[pass_] = \
                    {'growth_ratio_all' : growth_rate_all[num_iter - 1], \
                     'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_, \
                     'df_speciesAbun_next_pred' : y, \
                     'df_speciesAbun_next_obs' : x, \
                     'species_names' : species_names}
            # print(x.shape)
            
            id_ = np.where((x > 0) & (y > 0))[0]
            # id_ord = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # zero_thresh = -10
            y[y <= np.log10(thresh_zero)] = np.log10(thresh_zero)

            # if pass_ > 0:
            #     pass_tmp = pass_ - 1
            #     df_speciesAbun_ratio_tmp_1 = \
            #         df_speciesAbun_ratio_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
            #                                                  pass_tmp + 10]]
            #     x_r = \
            #         np.array(df_speciesAbun_ratio_tmp_1)
            #     x_new = -1 * np.ones(x_r.shape[0])
            #     for row_ in range(x_r.shape[0]):
            #         id_tmp = np.where(x_r[row_, :] > 0)[0]
            #         if len(id_tmp) > 0:
            #             x_new[row_] = 10**np.mean(np.log10(x_r[row_, id_tmp]))
            #     # id_tmp = np.where(x_new > 0)[0]
            #     # x_r = x_new[id_tmp]
            #     x_r = x_new
            #     x_r[x_r < 0] = 1 
            # else:
            #     x_r = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]])
            #     x_r = 10**np.mean(np.log10(x_r), axis=1).flatten()
            #     x_r /= np.array(df_speciesAbun_inoc).flatten()

            # # id_ = np.where((x > 0) & (y > 0))[0]
            # x_r = x_r[id_ord]

            # x_r[x_r > 1] = 3
            # x_r[x_r == 1] = 2
            # x_r[x_r < 1] = 1
            # growth_ord_tmp = [100*n**2 for n in x_r]

            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_] = {}
                RMSE_obj[pass_]["abundance"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted abundance (log scale)', fontsize=30)
                fig.supxlabel('observed abundance (log scale)', fontsize=30)
                # plt_ = sns.scatterplot(x=x, \
                #                        y=y, s=growth_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(x=x, \
                                    y=y, s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-8, 0], [-8, 0], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                # abs_mean_error = np.median(np.abs(y - x))
                std_error = np.sqrt(np.std(np.power(y - x, 2)))

                id_notzero_zero = np.where((x > -8) & (y <= -8))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                df_tmp = \
                    pd.DataFrame(data={"passage" : [pass_] * 8, 
                                    "p" : [p_tmp] * 8, 
                                    "metric_type" : ["corr_pearson_log", 
                                                        "corr_spearman", "slope", "slope_log", \
                                                        "corr_pearson_linear", \
                                                        "RMSE", \
                                                        "RSSE", "FNR"], 
                                    "metric" : [corr_val_pe_log[0], 
                                                corr_val_sp[0], slope, slope_log, \
                                                corr_val_pe[0], abs_mean_error, \
                                                std_error, \
                                                frac_zero],
                                    "pval" : [corr_val_pe_log[1], 
                                                corr_val_sp[1], slope_pval, slope_log_pval, \
                                                corr_val_pe[1], 0, 0, 0]})
                df_corr_slope = pd.concat([df_corr_slope, df_tmp], ignore_index=True)

                # plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
                plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="green", \
                        linewidth=3)

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed abundance at passage {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)

                save_dir = \
                    os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
            
            # Plot growth ratios
            if pass_ > 0:
                pass_tmp = pass_ - 1
                df_speciesAbun_ratio_tmp_1 = \
                    df_speciesAbun_ratio_mdl.iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]].copy()
                abun_prev = df_speciesAbun_prev_mdl.iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]].copy()
                abun_prev = np.array(abun_prev)
                abun_prev[abun_prev == 0] = thresh_zero
                abun_prev = 10**np.mean(np.log10(abun_prev), axis=1)
                x = \
                    np.array(df_speciesAbun_ratio_tmp_1)
                x_new = -1 * np.ones(x.shape[0])
                for row_ in range(x.shape[0]):
                    id_tmp = np.where(x[row_, :] > 0)[0]
                    if len(id_tmp) > 0:
                        x_new[row_] = 10**np.mean(np.log10(x[row_, id_tmp]))
                id_tmp = np.where(x_new > 0)[0]
                x = x_new[id_tmp]
                abun_prev = abun_prev.flatten()[id_tmp]
    #                 x = 10**(np.mean(np.log10(x), axis=1)).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                y = y[id_tmp]
            else:
                x = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]].copy())
                x[x == 0] = thresh_zero
                x = 10**np.mean(np.log10(x), axis=1).flatten()
                x_inoc_tmp = np.array(df_speciesAbun_inoc.copy()).flatten()
                x_inoc_tmp[x_inoc_tmp == 0] = thresh_zero
                x /= x_inoc_tmp
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                abun_prev = x_inoc_tmp

            y[y == 0] = 1e-8
            x[x == 0] = 1e-8
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                if pass_ > 0:
                    save_obj_return[pass_]['growth_ratio_obs'] = \
                        np.array(df_speciesAbun_ratio_tmp_1.copy())
                else:
                    save_obj_return[pass_]['growth_ratio_obs'] = np.hstack([x, x, x])
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # y=np.random.permutation(x)
            y[y <= -4] = -4
            x[x <= -4] = -4
            df_plt = pd.DataFrame(data={"x" : x, "y" : y, \
                                        "abun_prev" : abun_prev[id_]})
            prev_ord = np.argsort(abun_prev)
            prev_ord_tmp = [0.2*n**2 for n in prev_ord]
            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_]["growth_ratio"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted growth ratio (log scale)', fontsize=25)
                fig.supxlabel('observed growth ratio (log scale)', fontsize=25)
                # plt_ = sns.scatterplot(data=df_plt, x="x", \
                #                        y="y", s=prev_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(data=df_plt, x="x", \
                                    y="y", s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-4, 2], [-4, 2], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                std_error = np.sqrt(np.std(np.power(y - x, 2)))
                id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                # df_tmp = \
                #     pd.DataFrame(data={"p" : [p_tmp] * 8, 
                #                     "metric_type" : ["corr_pearson_log", 
                #                                         "corr_spearman", "slope", "slope_log", \
                #                                         "corr_pearson_linear", \
                #                                         "RMSE", \
                #                                         "std_error", "FNR"], 
                #                     "metric" : [corr_val_pe_log[0], 
                #                                 corr_val_sp[0], slope, slope_log, \
                #                                 corr_val_pe[0], abs_mean_error, \
                #                                 std_error, \
                #                                 frac_zero],
                #                     "pval" : [corr_val_pe_log[1], 
                #                                 corr_val_sp[1], slope_pval, slope_log_pval, \
                #                                 corr_val_pe[1], 0, 0, 0]})
                # df_corr_slope_growth = \
                #     pd.concat([df_corr_slope_growth, df_tmp], ignore_index=True)


                # plt_.plot([-4, 2], [np.log10(slope * (1e-4)), np.log10(slope * 1e2)], c="green")
                plt_.plot([-4, 2], [(slope_log * (-4)), (slope_log * 2)], c="green", \
                        linewidth=3)
                plt_.axhline(y=0, linestyle="dashed", c="black")
                plt_.axvline(x=0, linestyle="dashed", c="black")

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed growth ratio from {pass_} to {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)


                save_dir = \
                    os.path.abspath(os.path.join(dir_save_growth, \
                                                'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
                                                f'passage_{pass_}->{pass_ + 1}'))
                
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_growth_vs_' + \
                                                        f'observed_growth' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
    if plot_:
        for pass_ in range(num_passages):
            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.suptitle(f'corr, slope for predicted vs observed abundance \n' + \
                        f' passage {pass_ + 1}', \
                        fontsize=30)
            fig.supxlabel('p', fontsize=30)
            fig.supylabel('correlation or slope', fontsize=30)
            df_corr_slope_tmp = df_corr_slope.copy()
            df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
            
            df_corr_slope_tmp = \
                df_corr_slope_tmp[(df_corr_slope_tmp['metric_type'] != "corr_pearson_linear") & \
                                (df_corr_slope_tmp['metric_type'] != "slope")]
            plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_.set_xscale("log", base=10)

            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_stats.png'))
            fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)

    del df_corr_slope

    if return_sensitivity_ana:
        return save_obj_return, RMSE_obj
    # for pass_ in range(num_passages):
    #     fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
    #     fig.suptitle(f'corr, slope for predicted vs observed growth ratio \n' + \
    #                 f' passage {pass_ + 1}', \
    #                 fontsize=30)
    #     fig.supxlabel('p', fontsize=30)
    #     fig.supylabel('correlation or slope', fontsize=30)
    #     df_corr_slope_tmp = df_corr_slope_growth.copy()
    #     df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
    #     plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_.set_xscale("log", base=10)

    #     save_dir = \
    #         os.path.abspath(os.path.join(dir_save_growth, \
    #                                         'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
    #                                         f'passage_{pass_}->{pass_ + 1}'))
        
    #     if not os.path.exists(save_dir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_dir)
        
    #     file_save = os.path.abspath(os.path.join(save_dir, 
    #                                                 f'predicted_growth_vs_' + \
    #                                                 f'observed_growth' + \
    #                                                 f'_stats.png'))
    #     fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
    #     plt.close(fig.figure)


def blindly_pred_abun_growth_iter_Ri(p_vec_new, df_speciesMetab_cluster, \
                             df_speciesAbun_inoc, df_speciesAbun_mdl, \
                             df_speciesAbun_prev_mdl, \
                             df_speciesAbun_ratio_mdl, \
                             Ri_, dir_save_abun_obj, \
                             dir_save_abun, \
                             dir_save_growth, \
                             num_passages=6, num_iter=100, \
                             thresh_zero=1e-8, Ri_ss=True, plot_=True, \
                             save_data_obj=True, \
                             return_sensitivity_ana=False, \
                             get_prod=False, \
                             B_alone=None, \
                             df_speciesMetab_prod=None, \
                             prod_use_prev=True, \
                             num_passages_run=6, use_dilution=False, \
                             dilution_factor=15000, \
                             id_species_update=None, \
                             use_avg_for_prod=True, \
                             Ri_0=None):
    if Ri_0 is None:
        Ri_0 = Ri_.copy()
    num_species = df_speciesMetab_cluster.shape[0]
    # if id_species_update is None:
    #     id_species_update = np.arange(num_species)
    if id_species_update is not None:
        id_species_noupdate = \
            np.array(list(set(range(num_species)) - \
                 set(list(id_species_update))), dtype='long')
    # simulate inoculum abundances and initial growth ratios
    sample_names = df_speciesAbun_mdl.columns.values
    n_breps = 3
    samples_first = sample_names[0 + np.arange(n_breps) * num_passages]
    # ratio_init = np.zeros((num_species))
    # num_rand = 100
    # for rand_ in range(num_rand):
    #     ratio_init_rand = np.zeros((num_species, n_breps))
    #     for rep_ in range(n_breps):
    #         vals_ = np.random.exponential(1 / num_species, num_species)
    #         vals_ /= np.sum(vals_)
    #         ratio_init_rand[:, rep_] = df_speciesAbun_mdl[samples_first[rep_]].values / \
    #             vals_
    #     ratio_init += np.mean(np.log10(ratio_init_rand), axis=1)
    # ratio_init /= num_rand
    # ratio_init = 10**ratio_init
    ratio_init = np.ones(num_species)
    # print(ratio_init)


    df_corr_slope = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})

    df_corr_slope_growth = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})
    
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    if return_sensitivity_ana:
        save_obj_return = {}
        RMSE_obj = {}
    for count_p, p_tmp in enumerate(p_vec_new):
        for pass_ in range(num_passages_run):
            if pass_ == 0:
                df_speciesAbun_prev_tmp_ = \
                    df_speciesAbun_inoc.copy().iloc[:, :]
            else:
                growth_ratio_prev_ = growth_rate_all[num_iter - 1].copy()
                df_tmp = growth_rate_all[num_iter - 1].copy()
                for col_ in df_tmp.columns.values:
                    if id_species_update is not None:
                        df_tmp[col_][id_species_noupdate] = \
                            1
                    df_speciesAbun_prev_tmp_[col_] = \
                        df_tmp[col_].values * df_speciesAbun_prev_tmp_[col_].values

                    if id_species_update is None:
                        df_speciesAbun_prev_tmp_[col_] /= \
                            np.sum(df_speciesAbun_prev_tmp_[col_].values)
                    else:
                        df_speciesAbun_prev_tmp_[col_][id_species_update] /= \
                            np.sum(df_speciesAbun_prev_tmp_[col_].values[id_species_update])
            df_speciesAbun_next_tmp_ = df_speciesAbun_prev_tmp_.copy()


            growth_rate_all = {}
            growth_rate_all[0] = pd.DataFrame()
            growth_rate_tmp = pd.DataFrame()
            if Ri_ss:
                Ri_avg = Ri_.copy()
            else:
                if pass_ != 0:
                    Ri_avg = Ri_[count_p].copy()
                else:
                    Ri_avg = Ri_0[count_p].copy()
            
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                # growth_rate_all[0][col_] = np.ones((num_species))
                growth_rate_all[0][col_] = ratio_init
                growth_rate_tmp[col_] = np.ones((num_species))



            for iter_ in range(num_iter):
                if iter_ == 0:
                    iter_id = iter_
                else:
                    iter_id = iter_ - 1
                
                if (iter_ == 0) & (pass_ != 0):
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                        df_speciesAbun_next_tmp_.copy(), \
                                                        p_tmp, Ri_avg.copy(), \
                                                        # growth_ratio_prev_.copy(), \
                                                        growth_rate_all[iter_id].copy(), \
                                                        None, df_speciesMetab_tmp,
                                                        norm_=False, get_prod=get_prod, \
                                                        B_alone=B_alone, \
                                                        df_speciesMetab_prod=df_speciesMetab_prod, \
                                                        prod_use_prev=prod_use_prev, \
                                                        use_dilution=use_dilution, \
                                                        dilution_factor=dilution_factor, \
                                                        use_avg_for_prod=use_avg_for_prod)
                else:
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                           df_speciesAbun_next_tmp_.copy(), \
                                                           p_tmp, Ri_avg.copy(), \
                                                           growth_rate_all[iter_id].copy(), \
                                                           None, df_speciesMetab_tmp,
                                                           norm_=False, get_prod=get_prod, \
                                                           B_alone=B_alone, \
                                                           df_speciesMetab_prod=df_speciesMetab_prod, \
                                                           prod_use_prev=prod_use_prev, \
                                                           use_dilution=use_dilution, \
                                                           dilution_factor=dilution_factor, \
                                                           use_avg_for_prod=use_avg_for_prod)

                growth_rate_all[iter_] = df_growth_rate.copy()
            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun_obj, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             'predicted_abundance', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_Ri_fit' + \
                                                     f'_with_p{p_tmp}.pickle'))
            
            save_obj = {'growth_rate_all' : growth_rate_all, \
                        'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            # if return_sensitivity_ana:
            #     save_obj_return[pass_] = \
            #         {'growth_rate_all' : growth_rate_all[num_iter - 1], \
            #          'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            if save_data_obj:
                with open(file_save, "wb") as file_:
                    pickle.dump(save_obj, file_)  
            
            
            df_tmp = pd.DataFrame()
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                df_tmp[col_] = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values
                
                    
                if id_species_update is None:
                    growth_rate_all[num_iter - 1].copy()[col_] = \
                        growth_rate_all[num_iter - 1].copy()[col_].values / \
                        np.sum(df_tmp[col_].values)
                    # df_tmp[col_] /= np.sum(df_tmp[col_].values)
                else:
                    growth_rate_all[num_iter - 1].copy()[col_] = \
                        growth_rate_all[num_iter - 1].copy()[col_].values / \
                        np.sum(df_tmp[col_].values[id_species_update])
                    # df_tmp[col_][id_species_update] /= \
                    #     np.sum(df_tmp[col_].values[id_species_update])
                df_tmp[col_] = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values

            b_ = range(3)
            x = \
                np.array(df_speciesAbun_mdl.copy().iloc[:, [pass_, \
                                                            pass_ + num_passages, \
                                                            pass_ + 2 * num_passages]])
            x[x == 0] = thresh_zero
            # x = 10**(np.mean(np.log10(x), axis=1)).flatten()
            x = x.flatten()
            # y = np.array(df_tmp.copy())[:, :].flatten()
            y = np.array(df_tmp.copy())[:, :]
            y = np.hstack([y, y, y]).flatten()
            y[y == 0] = thresh_zero
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                save_obj_return[pass_] = \
                    {'growth_ratio_all' : growth_rate_all[num_iter - 1], \
                     'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_, \
                     'df_speciesAbun_next_pred' : y, \
                     'df_speciesAbun_next_obs' : x, \
                     'species_names' : species_names}
            # print(x.shape)
            
            id_ = np.where((x > 0) & (y > 0))[0]
            # id_ord = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            y[y <= -8] = -8

            # if pass_ > 0:
            #     pass_tmp = pass_ - 1
            #     df_speciesAbun_ratio_tmp_1 = \
            #         df_speciesAbun_ratio_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
            #                                                  pass_tmp + 10]]
            #     x_r = \
            #         np.array(df_speciesAbun_ratio_tmp_1)
            #     x_new = -1 * np.ones(x_r.shape[0])
            #     for row_ in range(x_r.shape[0]):
            #         id_tmp = np.where(x_r[row_, :] > 0)[0]
            #         if len(id_tmp) > 0:
            #             x_new[row_] = 10**np.mean(np.log10(x_r[row_, id_tmp]))
            #     # id_tmp = np.where(x_new > 0)[0]
            #     # x_r = x_new[id_tmp]
            #     x_r = x_new
            #     x_r[x_r < 0] = 1 
            # else:
            #     x_r = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]])
            #     x_r = 10**np.mean(np.log10(x_r), axis=1).flatten()
            #     x_r /= np.array(df_speciesAbun_inoc).flatten()

            # # id_ = np.where((x > 0) & (y > 0))[0]
            # x_r = x_r[id_ord]

            # x_r[x_r > 1] = 3
            # x_r[x_r == 1] = 2
            # x_r[x_r < 1] = 1
            # growth_ord_tmp = [100*n**2 for n in x_r]

            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_] = {}
                RMSE_obj[pass_]["abundance"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted abundance (log scale)', fontsize=30)
                fig.supxlabel('observed abundance (log scale)', fontsize=30)
                # plt_ = sns.scatterplot(x=x, \
                #                        y=y, s=growth_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(x=x, \
                                    y=y, s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-8, 0], [-8, 0], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                # abs_mean_error = np.median(np.abs(y - x))
                std_error = np.sqrt(np.std(np.power(y - x, 2)))

                id_notzero_zero = np.where((x > -8) & (y <= -8))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                df_tmp = \
                    pd.DataFrame(data={"passage" : [pass_] * 8, 
                                    "p" : [p_tmp] * 8, 
                                    "metric_type" : ["corr_pearson_log", 
                                                        "corr_spearman", "slope", "slope_log", \
                                                        "corr_pearson_linear", \
                                                        "RMSE", \
                                                        "RSSE", "FNR"], 
                                    "metric" : [corr_val_pe_log[0], 
                                                corr_val_sp[0], slope, slope_log, \
                                                corr_val_pe[0], abs_mean_error, \
                                                std_error, \
                                                frac_zero],
                                    "pval" : [corr_val_pe_log[1], 
                                                corr_val_sp[1], slope_pval, slope_log_pval, \
                                                corr_val_pe[1], 0, 0, 0]})
                df_corr_slope = pd.concat([df_corr_slope, df_tmp], ignore_index=True)

                # plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
                plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="green", \
                        linewidth=3)

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed abundance at passage {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)

                save_dir = \
                    os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
            
            # Plot growth ratios
            if pass_ > 0:
                pass_tmp = pass_ - 1
                df_speciesAbun_ratio_tmp_1 = \
                    df_speciesAbun_ratio_mdl.iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]].copy()
                abun_prev = df_speciesAbun_prev_mdl.iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]].copy()
                abun_prev = np.array(abun_prev)
                abun_prev[abun_prev == 0] = thresh_zero
                abun_prev = 10**np.mean(np.log10(abun_prev), axis=1)
                x = \
                    np.array(df_speciesAbun_ratio_tmp_1)
                x_new = -1 * np.ones(x.shape[0])
                for row_ in range(x.shape[0]):
                    id_tmp = np.where(x[row_, :] > 0)[0]
                    if len(id_tmp) > 0:
                        x_new[row_] = 10**np.mean(np.log10(x[row_, id_tmp]))
                id_tmp = np.where(x_new > 0)[0]
                x = x_new[id_tmp]
                abun_prev = abun_prev.flatten()[id_tmp]
    #                 x = 10**(np.mean(np.log10(x), axis=1)).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                y = y[id_tmp]
            else:
                x = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]].copy())
                x[x == 0] = thresh_zero
                x = 10**np.mean(np.log10(x), axis=1).flatten()
                x_inoc_tmp = np.array(df_speciesAbun_inoc.copy()).flatten()
                x_inoc_tmp[x_inoc_tmp == 0] = thresh_zero
                x /= x_inoc_tmp
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                abun_prev = x_inoc_tmp

            y[y == 0] = 1e-8
            x[x == 0] = 1e-8
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                if pass_ > 0:
                    save_obj_return[pass_]['growth_ratio_obs'] = \
                        np.array(df_speciesAbun_ratio_tmp_1.copy())
                else:
                    save_obj_return[pass_]['growth_ratio_obs'] = np.hstack([x, x, x])
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # y=np.random.permutation(x)
            y[y <= -4] = -4
            x[x <= -4] = -4
            df_plt = pd.DataFrame(data={"x" : x, "y" : y, \
                                        "abun_prev" : abun_prev[id_]})
            prev_ord = np.argsort(abun_prev)
            prev_ord_tmp = [0.2*n**2 for n in prev_ord]
            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_]["growth_ratio"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted growth ratio (log scale)', fontsize=25)
                fig.supxlabel('observed growth ratio (log scale)', fontsize=25)
                # plt_ = sns.scatterplot(data=df_plt, x="x", \
                #                        y="y", s=prev_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(data=df_plt, x="x", \
                                    y="y", s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-4, 2], [-4, 2], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                std_error = np.sqrt(np.std(np.power(y - x, 2)))
                id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                # df_tmp = \
                #     pd.DataFrame(data={"p" : [p_tmp] * 8, 
                #                     "metric_type" : ["corr_pearson_log", 
                #                                         "corr_spearman", "slope", "slope_log", \
                #                                         "corr_pearson_linear", \
                #                                         "RMSE", \
                #                                         "std_error", "FNR"], 
                #                     "metric" : [corr_val_pe_log[0], 
                #                                 corr_val_sp[0], slope, slope_log, \
                #                                 corr_val_pe[0], abs_mean_error, \
                #                                 std_error, \
                #                                 frac_zero],
                #                     "pval" : [corr_val_pe_log[1], 
                #                                 corr_val_sp[1], slope_pval, slope_log_pval, \
                #                                 corr_val_pe[1], 0, 0, 0]})
                # df_corr_slope_growth = \
                #     pd.concat([df_corr_slope_growth, df_tmp], ignore_index=True)


                # plt_.plot([-4, 2], [np.log10(slope * (1e-4)), np.log10(slope * 1e2)], c="green")
                plt_.plot([-4, 2], [(slope_log * (-4)), (slope_log * 2)], c="green", \
                        linewidth=3)
                plt_.axhline(y=0, linestyle="dashed", c="black")
                plt_.axvline(x=0, linestyle="dashed", c="black")

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed growth ratio from {pass_} to {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)


                save_dir = \
                    os.path.abspath(os.path.join(dir_save_growth, \
                                                'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
                                                f'passage_{pass_}->{pass_ + 1}'))
                
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_growth_vs_' + \
                                                        f'observed_growth' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
    if plot_:
        for pass_ in range(num_passages):
            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.suptitle(f'corr, slope for predicted vs observed abundance \n' + \
                        f' passage {pass_ + 1}', \
                        fontsize=30)
            fig.supxlabel('p', fontsize=30)
            fig.supylabel('correlation or slope', fontsize=30)
            df_corr_slope_tmp = df_corr_slope.copy()
            df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
            
            df_corr_slope_tmp = \
                df_corr_slope_tmp[(df_corr_slope_tmp['metric_type'] != "corr_pearson_linear") & \
                                (df_corr_slope_tmp['metric_type'] != "slope")]
            plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_.set_xscale("log", base=10)

            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_stats.png'))
            fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)

    del df_corr_slope

    if return_sensitivity_ana:
        return save_obj_return, RMSE_obj
    # for pass_ in range(num_passages):
    #     fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
    #     fig.suptitle(f'corr, slope for predicted vs observed growth ratio \n' + \
    #                 f' passage {pass_ + 1}', \
    #                 fontsize=30)
    #     fig.supxlabel('p', fontsize=30)
    #     fig.supylabel('correlation or slope', fontsize=30)
    #     df_corr_slope_tmp = df_corr_slope_growth.copy()
    #     df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
    #     plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_.set_xscale("log", base=10)

    #     save_dir = \
    #         os.path.abspath(os.path.join(dir_save_growth, \
    #                                         'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
    #                                         f'passage_{pass_}->{pass_ + 1}'))
        
    #     if not os.path.exists(save_dir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_dir)
        
    #     file_save = os.path.abspath(os.path.join(save_dir, 
    #                                                 f'predicted_growth_vs_' + \
    #                                                 f'observed_growth' + \
    #                                                 f'_stats.png'))
    #     fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
    #     plt.close(fig.figure)

def blindly_pred_abun_growth_without_inoc(p_vec_new, df_speciesMetab_cluster, \
                                          df_speciesAbun_inoc, df_speciesAbun_mdl, \
                                          df_speciesAbun_prev_mdl, \
                                          df_speciesAbun_ratio_mdl, \
                                          Ri_, dir_save_abun_obj, \
                                          dir_save_abun, \
                                          dir_save_growth, \
                                          num_passages=6, num_iter=100, \
                                          thresh_zero=1e-8, Ri_ss=True, plot_=True, \
                                          save_data_obj=True, \
                                          return_sensitivity_ana=False, num_brep=3, \
                                          metabs_cluster_id=None, \
                                          df_speciesAbun_mdl_true=None):
    num_species = df_speciesMetab_cluster.shape[0]

    # simulate inoculum abundances and initial growth ratios
    sample_names = df_speciesAbun_mdl.columns.values
    n_breps = num_brep
    # samples_first = sample_names[0 + np.arange(n_breps) * num_passages]
    ratio_init = np.zeros((num_species))
    # num_rand = 100
    # for rand_ in range(num_rand):
    #     ratio_init_rand = np.zeros((num_species, n_breps))
    #     for rep_ in range(n_breps):
    #         vals_ = np.random.exponential(1 / num_species, num_species)
    #         vals_ /= np.sum(vals_)
    #         ratio_init_rand[:, rep_] = df_speciesAbun_mdl[samples_first[rep_]].values / \
    #             vals_
    #     ratio_init += np.mean(np.log10(ratio_init_rand), axis=1)
    # ratio_init /= num_rand
    # ratio_init = 10**ratio_init
    ratio_init = np.ones(num_species)
    # print(ratio_init)

    pass_ = 0
    id_pass_init = []
    for rep_ in range(num_brep):
        id_pass_init += [pass_ + rep_ * num_passages]
    # id_pass_init = [pass_, pass_ + 6, pass_ + 12]


    df_corr_slope = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})

    df_corr_slope_growth = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    if return_sensitivity_ana:
        save_obj_return = {}
        RMSE_obj = {}
    for count_p, p_tmp in enumerate(p_vec_new):
        for pass_ in range(num_passages - 1):
            if pass_ == 0:
                df_speciesAbun_prev_tmp_ = \
                    df_speciesAbun_mdl.copy().iloc[:, id_pass_init]
                # print(df_speciesAbun_prev_tmp_)
            else:
                growth_ratio_prev_ = growth_rate_all[num_iter - 1].copy()
                df_tmp = growth_rate_all[num_iter - 1].copy()
                for col_ in df_tmp.columns.values:
                    df_speciesAbun_prev_tmp_[col_] = \
                        df_tmp[col_].values * df_speciesAbun_prev_tmp_[col_].values
                    # df_speciesAbun_prev_tmp_[col_] /= \
                    #     np.sum(df_speciesAbun_prev_tmp_[col_].values)
            df_speciesAbun_next_tmp_ = df_speciesAbun_prev_tmp_.copy()


            growth_rate_all = {}
            growth_rate_all[0] = pd.DataFrame()
            growth_rate_tmp = pd.DataFrame()
            if Ri_ss:
                Ri_avg = Ri_.copy()
            else:
                Ri_avg = Ri_[count_p].copy()
            
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                # growth_rate_all[0][col_] = np.ones((num_species))
                growth_rate_all[0][col_] = ratio_init
                growth_rate_tmp[col_] = np.ones((num_species))



            for iter_ in range(num_iter):
                if iter_ == 0:
                    iter_id = iter_
                else:
                    iter_id = iter_ - 1
                
                if (iter_ == 0) & (pass_ != 0):
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                        df_speciesAbun_next_tmp_.copy(), \
                                                        p_tmp, Ri_avg.copy(), \
                                                        # growth_ratio_prev_.copy(), \
                                                        growth_rate_all[iter_id].copy(), \
                                                        None, df_speciesMetab_tmp,
                                                        norm_=False, \
                                                        metabs_cluster_id=metabs_cluster_id)
                else:
                    df_growth_rate = \
                        compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                           df_speciesAbun_next_tmp_.copy(), \
                                                           p_tmp, Ri_avg.copy(), \
                                                           growth_rate_all[iter_id].copy(), \
                                                           None, df_speciesMetab_tmp,
                                                           norm_=False, \
                                                           metabs_cluster_id=metabs_cluster_id)

                growth_rate_all[iter_] = df_growth_rate.copy()

            
            save_obj = {'growth_rate_all' : growth_rate_all, \
                        'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_}
            if return_sensitivity_ana:
                save_obj_return[pass_] = \
                    {'growth_rate_all' : growth_rate_all[num_iter - 1].copy(), \
                     'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_.copy()}
                # if pass_ == 0:
                #     print(save_obj_return[pass_]['df_speciesAbun_prev'])
            if save_data_obj:
                save_dir = \
                    os.path.abspath(os.path.join(dir_save_abun_obj, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                'predicted_abundance', \
                                                f'passage_{pass_ + 2}'))
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_Ri_fit' + \
                                                        f'_with_p{p_tmp}.pickle'))
                with open(file_save, "wb") as file_:
                    pickle.dump(save_obj, file_)  
            
            
            df_tmp = pd.DataFrame()
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                df_tmp[col_] = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values
                df_tmp[col_] /= np.sum(df_tmp[col_].values)

            b_ = range(3)
            id_use = []
            for rep_ in range(num_brep):
                id_use.append(pass_ + 1 + rep_ * num_passages)
            # print(id_use)
            # print(df_speciesAbun_mdl)
            if df_speciesAbun_mdl_true is None:
                x = \
                    np.array(df_speciesAbun_mdl.iloc[:, id_use].copy())
            else:
                x = \
                    np.array(df_speciesAbun_mdl_true.iloc[:, id_use].copy())
            # x = 10**(np.mean(np.log10(x), axis=1)).flatten()
            x = x.flatten()
            y = np.array(df_tmp.copy())[:, :].flatten()
            y[y == 0] = thresh_zero
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                save_obj_return[pass_] = \
                    {'growth_ratio_all' : growth_rate_all[num_iter - 1], \
                     'df_speciesAbun_prev' : df_speciesAbun_prev_tmp_, \
                     'df_speciesAbun_next_pred' : y, \
                     'df_speciesAbun_next_obs' : x, \
                     'species_names' : species_names}

            id_ = np.where((x > 0) & (y > 0))[0]
            id_ord = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            y[y <= -8] = -8

            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_] = {}
                RMSE_obj[pass_]["abundance"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted abundance (log scale)', fontsize=30)
                fig.supxlabel('observed abundance (log scale)', fontsize=30)
                # plt_ = sns.scatterplot(x=x, \
                #                        y=y, s=growth_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(x=x, \
                                    y=y, s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-8, 0], [-8, 0], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                # abs_mean_error = np.median(np.abs(y - x))
                std_error = np.sqrt(np.std(np.power(y - x, 2)))

                id_notzero_zero = np.where((x > -8) & (y <= -8))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                df_tmp = \
                    pd.DataFrame(data={"passage" : [pass_] * 8, 
                                    "p" : [p_tmp] * 8, 
                                    "metric_type" : ["corr_pearson_log", 
                                                        "corr_spearman", "slope", "slope_log", \
                                                        "corr_pearson_linear", \
                                                        "RMSE", \
                                                        "RSSE", "FNR"], 
                                    "metric" : [corr_val_pe_log[0], 
                                                corr_val_sp[0], slope, slope_log, \
                                                corr_val_pe[0], abs_mean_error, \
                                                std_error, \
                                                frac_zero],
                                    "pval" : [corr_val_pe_log[1], 
                                                corr_val_sp[1], slope_pval, slope_log_pval, \
                                                corr_val_pe[1], 0, 0, 0]})
                df_corr_slope = pd.concat([df_corr_slope, df_tmp], ignore_index=True)

                # plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
                plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="green", \
                        linewidth=3)

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed abundance at passage {pass_ + 2}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)

                save_dir = \
                    os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 2}'))
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
            
            # Plot growth ratios
            if pass_ >= 0:
                pass_tmp = pass_
                id_use = []
                for rep_ in range(num_brep):
                    id_use.append(pass_tmp + rep_ * (num_passages - 1))
                df_speciesAbun_ratio_tmp_1 = \
                    df_speciesAbun_ratio_mdl.copy().iloc[:, id_use]
                abun_prev = df_speciesAbun_prev_mdl.copy().iloc[:, id_use]
                abun_prev = np.array(abun_prev)
                abun_prev = 10**np.mean(np.log10(abun_prev), axis=1)
                x = \
                    np.array(df_speciesAbun_ratio_tmp_1).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
            else:
                id_use = []
                for rep_ in range(num_brep):
                    id_use.append(pass_ + rep_ * (num_passages))
                x = np.array(df_speciesAbun_mdl.iloc[:, id_use])
                x = 10**np.mean(np.log10(x), axis=1).flatten()
                x /= np.array(df_speciesAbun_inoc).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                abun_prev = (np.array(df_speciesAbun_inoc).flatten())

            y[y == 0] = 1e-8
            if return_sensitivity_ana:
                species_names = df_speciesAbun_mdl.index.values
                species_names = np.hstack([species_names, species_names, species_names])
                if pass_ >= 0:
                    save_obj_return[pass_]['growth_ratio_obs'] = \
                        np.array(df_speciesAbun_ratio_tmp_1.copy())
                else:
                    save_obj_return[pass_]['growth_ratio_obs'] = np.hstack([x, x, x])
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # y=np.random.permutation(x)
            y[y <= -4] = -4
            abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
            if return_sensitivity_ana:
                RMSE_obj[pass_]["growth_ratio"] = abs_mean_error

            if plot_:
                fig, axes = plt.subplots(1, 1, figsize=(16, 18), sharey="row", sharex="col")
                fig.supylabel('predicted growth ratio (log scale)', fontsize=25)
                fig.supxlabel('observed growth ratio (log scale)', fontsize=25)
                # plt_ = sns.scatterplot(data=df_plt, x="x", \
                #                        y="y", s=prev_ord_tmp, ax=axes)
                plt_ = sns.scatterplot(x=x, \
                                    y=y, s=100, ax=axes)

                plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                                ax=axes)
                plt_.plot([-4, 2], [-4, 2], c="red", linewidth=3)

                corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
                corr_val_pe_log = scipy.stats.pearsonr(x, y)
                corr_val_sp = scipy.stats.spearmanr(x, y)
                std_error = np.sqrt(np.std(np.power(y - x, 2)))
                id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

                frac_zero = len(id_notzero_zero) / len(x)

                model = sm.OLS(10**y, 10**x).fit()
                slope = model.params[0]
                slope_pval = model.pvalues[0]
                
                model_log = sm.OLS(y, x).fit()
                slope_log = model_log.params[0]
                slope_log_pval = model_log.pvalues[0]

                # df_tmp = \
                #     pd.DataFrame(data={"p" : [p_tmp] * 8, 
                #                     "metric_type" : ["corr_pearson_log", 
                #                                         "corr_spearman", "slope", "slope_log", \
                #                                         "corr_pearson_linear", \
                #                                         "RMSE", \
                #                                         "std_error", "FNR"], 
                #                     "metric" : [corr_val_pe_log[0], 
                #                                 corr_val_sp[0], slope, slope_log, \
                #                                 corr_val_pe[0], abs_mean_error, \
                #                                 std_error, \
                #                                 frac_zero],
                #                     "pval" : [corr_val_pe_log[1], 
                #                                 corr_val_sp[1], slope_pval, slope_log_pval, \
                #                                 corr_val_pe[1], 0, 0, 0]})
                # df_corr_slope_growth = \
                #     pd.concat([df_corr_slope_growth, df_tmp], ignore_index=True)


                # plt_.plot([-4, 2], [np.log10(slope * (1e-4)), np.log10(slope * 1e2)], c="green")
                plt_.plot([-4, 2], [(slope_log * (-4)), (slope_log * 2)], c="green", \
                        linewidth=3)
                plt_.axhline(y=0, linestyle="dashed", c="black")
                plt_.axvline(x=0, linestyle="dashed", c="black")

                # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe[1]) + \
                #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
                #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                #                 '{:.3e}'.format(corr_val_sp[1]) + \
                #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_pval) + \
                #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                #                 '{:.3e}'.format(slope_log_pval) + \
                #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                #         f'\n fit with p = {p_tmp}'
                title_ = f'predicted vs observed growth ratio from {pass_} to {pass_ + 1}' + \
                        f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_pe_log[1]) + \
                        f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                                '{:.3e}'.format(corr_val_sp[1]) + \
                        f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                                '{:.3e}'.format(slope_log_pval) + \
                        f'\n RMSE = {np.round(abs_mean_error, 3)}'

                axes.set_title(title_, size=25)
                plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=20)
                plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=20)


                save_dir = \
                    os.path.abspath(os.path.join(dir_save_growth, \
                                                'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}->{pass_ + 2}'))
                
                if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                    os.makedirs(save_dir)
                
                file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_growth_vs_' + \
                                                        f'observed_growth' + \
                                                        f'_with_p{p_tmp}.png'))

                fig.figure.savefig(file_save, \
                                dpi=300, transparent=False, facecolor="white")
                plt.close(fig.figure)
    if plot_:
        for pass_ in range(num_passages):
            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.suptitle(f'corr, slope for predicted vs observed abundance \n' + \
                        f' passage {pass_ + 1}', \
                        fontsize=30)
            fig.supxlabel('p', fontsize=30)
            fig.supylabel('correlation or slope', fontsize=30)
            df_corr_slope_tmp = df_corr_slope.copy()
            df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
            
            df_corr_slope_tmp = \
                df_corr_slope_tmp[(df_corr_slope_tmp['metric_type'] != "corr_pearson_linear") & \
                                (df_corr_slope_tmp['metric_type'] != "slope")]
            plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
                                hue="metric_type", ax=axes)
            plt_.set_xscale("log", base=10)

            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun, \
                                                'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                                f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
                # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                        f'predicted_abundance_vs_' + \
                                                        f'observed_abundance' + \
                                                        f'_stats.png'))
            fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)

    del df_corr_slope

    if return_sensitivity_ana:
        return save_obj_return, RMSE_obj
    # for pass_ in range(num_passages):
    #     fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
    #     fig.suptitle(f'corr, slope for predicted vs observed growth ratio \n' + \
    #                 f' passage {pass_ + 1}', \
    #                 fontsize=30)
    #     fig.supxlabel('p', fontsize=30)
    #     fig.supylabel('correlation or slope', fontsize=30)
    #     df_corr_slope_tmp = df_corr_slope_growth.copy()
    #     df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
    #     plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_.set_xscale("log", base=10)

    #     save_dir = \
    #         os.path.abspath(os.path.join(dir_save_growth, \
    #                                         'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
    #                                         f'passage_{pass_}->{pass_ + 1}'))
        
    #     if not os.path.exists(save_dir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_dir)
        
    #     file_save = os.path.abspath(os.path.join(save_dir, 
    #                                                 f'predicted_growth_vs_' + \
    #                                                 f'observed_growth' + \
    #                                                 f'_stats.png'))
    #     fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
    #     plt.close(fig.figure)

def blindly_pred_abun_loggrowth(p_vec_new, df_speciesMetab_cluster, \
                             df_speciesAbun_inoc, df_speciesAbun_mdl, \
                             df_speciesAbun_prev_mdl, \
                             df_speciesAbun_ratio_mdl, \
                             Ri_, dir_save_abun_obj, \
                             dir_save_abun, \
                             dir_save_growth, \
                             num_passages=6, num_iter=100, \
                             thresh_zero=1e-8, Ri_ss=True):
    num_species = df_speciesMetab_cluster.shape[0]
    df_corr_slope = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})

    df_corr_slope_growth = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    for count_p, p_tmp in enumerate(p_vec_new):
        for pass_ in range(num_passages):
            if pass_ == 0:
                df_speciesAbun_prev_tmp_ = \
                    df_speciesAbun_inoc.copy().iloc[:, :]
            else:
                growth_ratio_prev_ = growth_rate_all[num_iter - 1].copy()
                df_tmp = growth_rate_all[num_iter - 1].copy()
                for col_ in df_tmp.columns.values:
                    df_speciesAbun_prev_tmp_[col_] = \
                        10**df_tmp[col_].values * df_speciesAbun_prev_tmp_[col_].values
            df_speciesAbun_next_tmp_ = df_speciesAbun_prev_tmp_.copy()


            growth_rate_all = {}
            growth_rate_all[0] = pd.DataFrame()
            growth_rate_tmp = pd.DataFrame()
            if Ri_ss:
                Ri_avg = Ri_.copy()
            else:
                Ri_avg = Ri_[count_p].copy()
            
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                growth_rate_all[0][col_] = np.zeros((num_species))
                growth_rate_tmp[col_] = np.zeros((num_species))

            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.supylabel('predicted abundance (log scale)', fontsize=25)
            fig.supxlabel('observed abundance (log scale)', fontsize=25)

            for iter_ in range(num_iter):
                if iter_ == 0:
                    iter_id = iter_
                else:
                    iter_id = iter_ - 1
                
                if (iter_ == 0) & (pass_ != 0):
                    df_growth_rate = \
                        compute_log_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                        df_speciesAbun_next_tmp_.copy(), \
                                                        p_tmp, Ri_avg.copy(), \
                                                        # growth_ratio_prev_.copy(), \
                                                        growth_rate_all[iter_id].copy(), \
                                                        None, df_speciesMetab_tmp,
                                                        norm_=False)
                else:
                    df_growth_rate = \
                        compute_log_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
                                                           df_speciesAbun_next_tmp_.copy(), \
                                                           p_tmp, Ri_avg.copy(), \
                                                           growth_rate_all[iter_id].copy(), \
                                                           None, df_speciesMetab_tmp,
                                                           norm_=False)

                growth_rate_all[iter_] = df_growth_rate.copy()
            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun_obj, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             'predicted_abundance', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_Ri_fit_steadyState' + \
                                                     f'_with_p{p_tmp}.pickle'))

            with open(file_save, "wb") as file_:
                pickle.dump(growth_rate_all, file_)  
            
            df_tmp = pd.DataFrame()
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                df_tmp[col_] = \
                    10**growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values

            b_ = range(3)
            x = \
                np.array(df_speciesAbun_mdl.copy().iloc[:, [pass_, \
                                                            pass_ + num_passages, \
                                                            pass_ + 2 * num_passages]])
            x = 10**(np.mean(np.log10(x), axis=1)).flatten()
            y = 10**np.array(df_tmp.copy())[:, :].flatten()
            y[y == 0] = thresh_zero
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            y[y <= -8] = -8

            plt_ = sns.scatterplot(x=x, \
                                y=y, ax=axes)

            plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                            ax=axes)
            plt_.plot([-8, 0], [-8, 0], c="red")

            corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
            corr_val_pe_log = scipy.stats.pearsonr(x, y)
            corr_val_sp = scipy.stats.spearmanr(x, y)
            abs_mean_error = np.median(np.abs(y - x))
            std_error = np.std(np.abs(y - x))

            id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

            frac_zero = len(id_notzero_zero) / len(x)

            model = sm.OLS(10**y, 10**x).fit()
            slope = model.params[0]
            slope_pval = model.pvalues[0]
            
            model_log = sm.OLS(y, x).fit()
            slope_log = model_log.params[0]
            slope_log_pval = model_log.pvalues[0]

            df_tmp = \
                pd.DataFrame(data={"passage" : [pass_] * 8, 
                                "p" : [p_tmp] * 8, 
                                "metric_type" : ["corr_pearson_log", 
                                                    "corr_spearman", "slope", "slope_log", \
                                                    "corr_pearson_linear", \
                                                    "abs_median_error", \
                                                    "std_error", "FNR"], 
                                "metric" : [corr_val_pe_log[0], 
                                            corr_val_sp[0], slope, slope_log, \
                                            corr_val_pe[0], abs_mean_error, \
                                            std_error, \
                                            frac_zero],
                                "pval" : [corr_val_pe_log[1], 
                                            corr_val_sp[1], slope_pval, slope_log_pval, \
                                            corr_val_pe[1], 0, 0, 0]})
            df_corr_slope = pd.concat([df_corr_slope, df_tmp], ignore_index=True)

            plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
            plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="blue")

            title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe[1]) + \
                    f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe_log[1]) + \
                    f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_sp[1]) + \
                    f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_pval) + \
                    f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_log_pval) + \
                    f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                    f'\n fit with p = {p_tmp}'

            axes.set_title(title_, size=15)

            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_vs_' + \
                                                     f'observed_abundance' + \
                                                     f'_with_p{p_tmp}.png'))

            fig.figure.savefig(file_save, \
                               dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)
            
            # Plot growth ratios
            
            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.supylabel('predicted growth ratio (log scale)', fontsize=25)
            fig.supxlabel('observed growth ratio (log scale)', fontsize=25)
            if pass_ > 0:
                pass_tmp = pass_ - 1
                df_speciesAbun_ratio_tmp_1 = \
                    df_speciesAbun_ratio_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]]
                abun_prev = df_speciesAbun_prev_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]]
                abun_prev = np.array(abun_prev)
                abun_prev = 10**np.mean(np.log10(abun_prev), axis=1)
                x = \
                    np.array(df_speciesAbun_ratio_tmp_1)
                x_new = -1 * np.ones(x.shape[0])
                for row_ in range(x.shape[0]):
                    id_tmp = np.where(x[row_, :] > 0)[0]
                    if len(id_tmp) > 0:
                        x_new[row_] = 10**np.mean(np.log10(x[row_, id_tmp]))
                id_tmp = np.where(x_new > 0)[0]
                x = x_new[id_tmp]
                abun_prev = abun_prev.flatten()[id_tmp]
    #                 x = 10**(np.mean(np.log10(x), axis=1)).flatten()
                y = 10**np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                y = y[id_tmp]
            else:
                x = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]])
                x = 10**np.mean(np.log10(x), axis=1).flatten()
                x /= np.array(df_speciesAbun_inoc).flatten()
                y = 10**np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                abun_prev = (np.array(df_speciesAbun_inoc).flatten())

            y[y == 0] = 1e-8
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # y=np.random.permutation(x)
            y[y <= -8] = -8
            df_plt = pd.DataFrame(data={"x" : x, "y" : y, \
                                        "abun_prev" : abun_prev})
            prev_ord = np.argsort(abun_prev)
            prev_ord_tmp = [0.2*n**2 for n in prev_ord]
            plt_ = sns.scatterplot(data=df_plt, x="x", \
                                   y="y", s=prev_ord_tmp, ax=axes)

            plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                            ax=axes)
            plt_.plot([-4, 2], [-4, 2], c="red")

            corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
            corr_val_pe_log = scipy.stats.pearsonr(x, y)
            corr_val_sp = scipy.stats.spearmanr(x, y)
            abs_mean_error = np.median(np.abs(y - x))
            std_error = np.std(np.abs(y - x))

            id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

            frac_zero = len(id_notzero_zero) / len(x)

            model = sm.OLS(10**y, 10**x).fit()
            slope = model.params[0]
            slope_pval = model.pvalues[0]
            
            model_log = sm.OLS(y, x).fit()
            slope_log = model_log.params[0]
            slope_log_pval = model_log.pvalues[0]

            df_tmp = \
                pd.DataFrame(data={"p" : [p_tmp] * 8, 
                                "metric_type" : ["corr_pearson_log", 
                                                    "corr_spearman", "slope", "slope_log", \
                                                    "corr_pearson_linear", \
                                                    "abs_median_error", \
                                                    "std_error", "FNR"], 
                                "metric" : [corr_val_pe_log[0], 
                                            corr_val_sp[0], slope, slope_log, \
                                            corr_val_pe[0], abs_mean_error, \
                                            std_error, \
                                            frac_zero],
                                "pval" : [corr_val_pe_log[1], 
                                            corr_val_sp[1], slope_pval, slope_log_pval, \
                                            corr_val_pe[1], 0, 0, 0]})
            df_corr_slope_growth = \
                pd.concat([df_corr_slope_growth, df_tmp], ignore_index=True)


            plt_.plot([-4, 2], [np.log10(slope * (1e-4)), np.log10(slope * 1e2)], c="green")
            plt_.plot([-4, 2], [(slope_log * (-4)), (slope_log * 2)], c="blue")
            plt_.axhline(y=0, linestyle="dashed", c="black")
            plt_.axvline(x=0, linestyle="dashed", c="black")

            title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe[1]) + \
                    f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe_log[1]) + \
                    f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_sp[1]) + \
                    f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_pval) + \
                    f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_log_pval) + \
                    f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                    f'\n fit with p = {p_tmp}'

            axes.set_title(title_, size=15)


            save_dir = \
                os.path.abspath(os.path.join(dir_save_growth, \
                                             'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
                                             f'passage_{pass_}->{pass_ + 1}'))
            
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_growth_vs_' + \
                                                     f'observed_growth' + \
                                                     f'_with_p{p_tmp}.png'))

            fig.figure.savefig(file_save, \
                               dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)
    for pass_ in range(num_passages):
        fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
        fig.suptitle(f'corr, slope for predicted vs observed abundance \n' + \
                    f' passage {pass_ + 1}', \
                    fontsize=30)
        fig.supxlabel('p', fontsize=30)
        fig.supylabel('correlation or slope', fontsize=30)
        df_corr_slope_tmp = df_corr_slope.copy()
        df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
        plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
                            hue="metric_type", ax=axes)
        plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
                            hue="metric_type", ax=axes)
        plt_.set_xscale("log", base=10)

        save_dir = \
            os.path.abspath(os.path.join(dir_save_abun, \
                                            'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                            f'passage_{pass_ + 1}'))
        if not os.path.exists(save_dir):
            # Create a new directory because it does not exist
            os.makedirs(save_dir)
        
        file_save = os.path.abspath(os.path.join(save_dir, 
                                                    f'predicted_abundance_vs_' + \
                                                    f'observed_abundance' + \
                                                    f'_stats.png'))
        fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
        plt.close(fig.figure)

    del df_corr_slope
    # for pass_ in range(num_passages):
    #     fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
    #     fig.suptitle(f'corr, slope for predicted vs observed growth ratio \n' + \
    #                 f' passage {pass_ + 1}', \
    #                 fontsize=30)
    #     fig.supxlabel('p', fontsize=30)
    #     fig.supylabel('correlation or slope', fontsize=30)
    #     df_corr_slope_tmp = df_corr_slope_growth.copy()
    #     df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
    #     plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_.set_xscale("log", base=10)

    #     save_dir = \
    #         os.path.abspath(os.path.join(dir_save_growth, \
    #                                         'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
    #                                         f'passage_{pass_}->{pass_ + 1}'))
        
    #     if not os.path.exists(save_dir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_dir)
        
    #     file_save = os.path.abspath(os.path.join(save_dir, 
    #                                                 f'predicted_growth_vs_' + \
    #                                                 f'observed_growth' + \
    #                                                 f'_stats.png'))
    #     fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
    #     plt.close(fig.figure)

def blindly_pred_abun_growth_seq(p_vec_new, df_speciesMetab_cluster, \
                                 df_speciesAbun_inoc, df_speciesAbun_mdl, \
                                 df_speciesAbun_prev_mdl, \
                                 df_speciesAbun_ratio_mdl, \
                                 Ri_, dir_save_abun_obj, \
                                 dir_save_abun, \
                                 dir_save_growth, \
                                 num_passages=6, num_iter=100, \
                                 thresh_zero=1e-8, Ri_ss=True):
    num_species = df_speciesMetab_cluster.shape[0]
    df_corr_slope = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})

    df_corr_slope_growth = pd.DataFrame(data={"norm_status" : [],
                                    "passage" : [], 
                                    "p" : [],
                                    "metric_type" : [], 
                                    "metric" : [],
                                    "pval" : []})
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    for count_p, p_tmp in enumerate(p_vec_new):
        for pass_ in range(num_passages):
            if pass_ == 0:
                df_speciesAbun_prev_tmp_ = \
                    df_speciesAbun_inoc.copy().iloc[:, :]
            else:
                growth_ratio_prev_ = growth_rate_all[num_iter - 1].copy()
                df_tmp = growth_rate_all[num_iter - 1].copy()
                for col_ in df_tmp.columns.values:
                    df_speciesAbun_prev_tmp_[col_] = \
                        df_tmp[col_].values * df_speciesAbun_prev_tmp_[col_].values
            df_speciesAbun_next_tmp_ = df_speciesAbun_prev_tmp_.copy()


            growth_rate_all = {}
            growth_rate_all[0] = pd.DataFrame()
            growth_rate_tmp = pd.DataFrame()
            if Ri_ss:
                Ri_avg = Ri_.copy()
            else:
                Ri_avg = Ri_[count_p].copy()
            
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                growth_rate_all[0][col_] = np.ones((num_species))
                growth_rate_tmp[col_] = np.ones((num_species))

            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.supylabel('predicted abundance (log scale)', fontsize=25)
            fig.supxlabel('observed abundance (log scale)', fontsize=25)

            growth_rate_all =  \
                compute_growth_ratio_iterate_blind_seq(df_speciesAbun_prev_tmp_, \
                                                       df_speciesAbun_next_tmp_, p_tmp, \
                                                       Ri_avg.copy(), \
                                                       df_speciesMetab_tmp, num_iter=num_iter)

            # for iter_ in range(num_iter):
            #     if iter_ == 0:
            #         iter_id = iter_
            #     else:
            #         iter_id = iter_ - 1
                
            #     if (iter_ == 0) & (pass_ != 0):
            #         df_growth_rate = \
            #             compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
            #                                             df_speciesAbun_next_tmp_.copy(), \
            #                                             p_tmp, Ri_avg.copy(), \
            #                                             # growth_ratio_prev_.copy(), \
            #                                             growth_rate_all[iter_id].copy(), \
            #                                             None, df_speciesMetab_tmp,
            #                                             norm_=False)
            #     else:
            #         df_growth_rate = \
            #             compute_growth_ratio_iterate_blind(df_speciesAbun_prev_tmp_.copy(), \
            #                                                df_speciesAbun_next_tmp_.copy(), \
            #                                                p_tmp, Ri_avg.copy(), \
            #                                                growth_rate_all[iter_id].copy(), \
            #                                                None, df_speciesMetab_tmp,
            #                                                norm_=False)

            #     growth_rate_all[iter_] = df_growth_rate.copy()
            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun_obj, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             'predicted_abundance', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_Ri_fit_steadyState' + \
                                                     f'_with_p{p_tmp}.pickle'))

            with open(file_save, "wb") as file_:
                pickle.dump(growth_rate_all, file_)  
            
            df_tmp = pd.DataFrame()
            for col_ in df_speciesAbun_prev_tmp_.columns.values:
                df_tmp[col_] = \
                    growth_rate_all[num_iter - 1].copy()[col_].values * \
                    df_speciesAbun_prev_tmp_[col_].values

            b_ = range(3)
            x = \
                np.array(df_speciesAbun_mdl.copy().iloc[:, [pass_, \
                                                            pass_ + num_passages, \
                                                            pass_ + 2 * num_passages]])
            x = 10**(np.mean(np.log10(x), axis=1)).flatten()
            y = np.array(df_tmp.copy())[:, :].flatten()
            y[y == 0] = thresh_zero
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            y[y <= -8] = -8

            plt_ = sns.scatterplot(x=x, \
                                y=y, ax=axes)

            plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                            ax=axes)
            plt_.plot([-8, 0], [-8, 0], c="red")

            corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
            corr_val_pe_log = scipy.stats.pearsonr(x, y)
            corr_val_sp = scipy.stats.spearmanr(x, y)
            abs_mean_error = np.median(np.abs(y - x))
            std_error = np.std(np.abs(y - x))

            id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

            frac_zero = len(id_notzero_zero) / len(x)

            model = sm.OLS(10**y, 10**x).fit()
            slope = model.params[0]
            slope_pval = model.pvalues[0]
            
            model_log = sm.OLS(y, x).fit()
            slope_log = model_log.params[0]
            slope_log_pval = model_log.pvalues[0]

            df_tmp = \
                pd.DataFrame(data={"passage" : [pass_] * 8, 
                                "p" : [p_tmp] * 8, 
                                "metric_type" : ["corr_pearson_log", 
                                                    "corr_spearman", "slope", "slope_log", \
                                                    "corr_pearson_linear", \
                                                    "abs_median_error", \
                                                    "std_error", "FNR"], 
                                "metric" : [corr_val_pe_log[0], 
                                            corr_val_sp[0], slope, slope_log, \
                                            corr_val_pe[0], abs_mean_error, \
                                            std_error, \
                                            frac_zero],
                                "pval" : [corr_val_pe_log[1], 
                                            corr_val_sp[1], slope_pval, slope_log_pval, \
                                            corr_val_pe[1], 0, 0, 0]})
            df_corr_slope = pd.concat([df_corr_slope, df_tmp], ignore_index=True)

            plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
            plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="blue")

            title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe[1]) + \
                    f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe_log[1]) + \
                    f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_sp[1]) + \
                    f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_pval) + \
                    f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_log_pval) + \
                    f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                    f'\n fit with p = {p_tmp}'

            axes.set_title(title_, size=15)

            save_dir = \
                os.path.abspath(os.path.join(dir_save_abun, \
                                             'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                             f'passage_{pass_ + 1}'))
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_abundance_vs_' + \
                                                     f'observed_abundance' + \
                                                     f'_with_p{p_tmp}.png'))

            fig.figure.savefig(file_save, \
                               dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)
            
            # Plot growth ratios
            
            fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
            fig.supylabel('predicted growth ratio (log scale)', fontsize=25)
            fig.supxlabel('observed growth ratio (log scale)', fontsize=25)
            if pass_ > 0:
                pass_tmp = pass_ - 1
                df_speciesAbun_ratio_tmp_1 = \
                    df_speciesAbun_ratio_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]]
                abun_prev = df_speciesAbun_prev_mdl.copy().iloc[:, [pass_tmp, pass_tmp + 5, \
                                                             pass_tmp + 10]]
                abun_prev = np.array(abun_prev)
                abun_prev = 10**np.mean(np.log10(abun_prev), axis=1)
                x = \
                    np.array(df_speciesAbun_ratio_tmp_1)
                x_new = -1 * np.ones(x.shape[0])
                for row_ in range(x.shape[0]):
                    id_tmp = np.where(x[row_, :] > 0)[0]
                    if len(id_tmp) > 0:
                        x_new[row_] = 10**np.mean(np.log10(x[row_, id_tmp]))
                id_tmp = np.where(x_new > 0)[0]
                x = x_new[id_tmp]
                abun_prev = abun_prev.flatten()[id_tmp]
    #                 x = 10**(np.mean(np.log10(x), axis=1)).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                y = y[id_tmp]
            else:
                x = np.array(df_speciesAbun_mdl.iloc[:, [pass_, pass_ + 6, pass_ + 12]])
                x = 10**np.mean(np.log10(x), axis=1).flatten()
                x /= np.array(df_speciesAbun_inoc).flatten()
                y = np.array(growth_rate_all[num_iter - 1].copy())[:, :].flatten()
                abun_prev = (np.array(df_speciesAbun_inoc).flatten())

            y[y == 0] = 1e-8
            id_ = np.where((x > 0) & (y > 0))[0]
            x = np.log10(x[id_])
            y = np.log10(y[id_])
            # y=np.random.permutation(x)
            y[y <= -8] = -8
            df_plt = pd.DataFrame(data={"x" : x, "y" : y, \
                                        "abun_prev" : abun_prev})
            prev_ord = np.argsort(abun_prev)
            prev_ord_tmp = [0.2*n**2 for n in prev_ord]
            plt_ = sns.scatterplot(data=df_plt, x="x", \
                                   y="y", s=prev_ord_tmp, ax=axes)

            plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                            ax=axes)
            plt_.plot([-4, 2], [-4, 2], c="red")

            corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
            corr_val_pe_log = scipy.stats.pearsonr(x, y)
            corr_val_sp = scipy.stats.spearmanr(x, y)
            abs_mean_error = np.median(np.abs(y - x))
            std_error = np.std(np.abs(y - x))

            id_notzero_zero = np.where((x > -5) & (y <= -5))[0]

            frac_zero = len(id_notzero_zero) / len(x)

            model = sm.OLS(10**y, 10**x).fit()
            slope = model.params[0]
            slope_pval = model.pvalues[0]
            
            model_log = sm.OLS(y, x).fit()
            slope_log = model_log.params[0]
            slope_log_pval = model_log.pvalues[0]

            df_tmp = \
                pd.DataFrame(data={"p" : [p_tmp] * 8, 
                                "metric_type" : ["corr_pearson_log", 
                                                    "corr_spearman", "slope", "slope_log", \
                                                    "corr_pearson_linear", \
                                                    "abs_median_error", \
                                                    "std_error", "FNR"], 
                                "metric" : [corr_val_pe_log[0], 
                                            corr_val_sp[0], slope, slope_log, \
                                            corr_val_pe[0], abs_mean_error, \
                                            std_error, \
                                            frac_zero],
                                "pval" : [corr_val_pe_log[1], 
                                            corr_val_sp[1], slope_pval, slope_log_pval, \
                                            corr_val_pe[1], 0, 0, 0]})
            df_corr_slope_growth = \
                pd.concat([df_corr_slope_growth, df_tmp], ignore_index=True)


            plt_.plot([-4, 2], [np.log10(slope * (1e-4)), np.log10(slope * 1e2)], c="green")
            plt_.plot([-4, 2], [(slope_log * (-4)), (slope_log * 2)], c="blue")
            plt_.axhline(y=0, linestyle="dashed", c="black")
            plt_.axvline(x=0, linestyle="dashed", c="black")

            title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe[1]) + \
                    f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_pe_log[1]) + \
                    f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                            '{:.3e}'.format(corr_val_sp[1]) + \
                    f'\n slope = {np.round(slope, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_pval) + \
                    f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                            '{:.3e}'.format(slope_log_pval) + \
                    f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
                    f'\n fit with p = {p_tmp}'

            axes.set_title(title_, size=15)


            save_dir = \
                os.path.abspath(os.path.join(dir_save_growth, \
                                             'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
                                             f'passage_{pass_}->{pass_ + 1}'))
            
            if not os.path.exists(save_dir):
               # Create a new directory because it does not exist
                os.makedirs(save_dir)
            
            file_save = os.path.abspath(os.path.join(save_dir, 
                                                     f'predicted_growth_vs_' + \
                                                     f'observed_growth' + \
                                                     f'_with_p{p_tmp}.png'))

            fig.figure.savefig(file_save, \
                               dpi=300, transparent=False, facecolor="white")
            plt.close(fig.figure)
    for pass_ in range(num_passages):
        fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
        fig.suptitle(f'corr, slope for predicted vs observed abundance \n' + \
                    f' passage {pass_ + 1}', \
                    fontsize=30)
        fig.supxlabel('p', fontsize=30)
        fig.supylabel('correlation or slope', fontsize=30)
        df_corr_slope_tmp = df_corr_slope.copy()
        df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
        plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
                            hue="metric_type", ax=axes)
        plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
                            hue="metric_type", ax=axes)
        plt_.set_xscale("log", base=10)

        save_dir = \
            os.path.abspath(os.path.join(dir_save_abun, \
                                            'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                            f'passage_{pass_ + 1}'))
        if not os.path.exists(save_dir):
            # Create a new directory because it does not exist
            os.makedirs(save_dir)
        
        file_save = os.path.abspath(os.path.join(save_dir, 
                                                    f'predicted_abundance_vs_' + \
                                                    f'observed_abundance' + \
                                                    f'_stats.png'))
        fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
        plt.close(fig.figure)

    del df_corr_slope
    # for pass_ in range(num_passages):
    #     fig, axes = plt.subplots(1, 1, figsize=(20, 14), sharey="row", sharex="col")
    #     fig.suptitle(f'corr, slope for predicted vs observed growth ratio \n' + \
    #                 f' passage {pass_ + 1}', \
    #                 fontsize=30)
    #     fig.supxlabel('p', fontsize=30)
    #     fig.supylabel('correlation or slope', fontsize=30)
    #     df_corr_slope_tmp = df_corr_slope_growth.copy()
    #     df_corr_slope_tmp = df_corr_slope_tmp[df_corr_slope_tmp["passage"] == pass_]
    #     plt_ = sns.scatterplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_ = sns.lineplot(data=df_corr_slope_tmp, x="p", y="metric", \
    #                         hue="metric_type", ax=axes)
    #     plt_.set_xscale("log", base=10)

    #     save_dir = \
    #         os.path.abspath(os.path.join(dir_save_growth, \
    #                                         'predicted_vs_observed_growth_ratio_LeaveOneOutRi', \
    #                                         f'passage_{pass_}->{pass_ + 1}'))
        
    #     if not os.path.exists(save_dir):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_dir)
        
    #     file_save = os.path.abspath(os.path.join(save_dir, 
    #                                                 f'predicted_growth_vs_' + \
    #                                                 f'observed_growth' + \
    #                                                 f'_stats.png'))
    #     fig.figure.savefig(file_save, dpi=300, transparent=False, facecolor="white")
    #     plt.close(fig.figure)

def fit_dynamic_Ri_no_f(df_speciesMetab_cluster, \
                  df_speciesAbun_prev_mdl, df_speciesAbun_next_mdl, \
                  df_speciesAbun_ratio_mdl, p_vec_new, \
                  file_save, num_passages=5, pass_rm=[0, 1, 2], \
                  save_data=True, verbose=True, method="linear", alpha=0, \
                  use_loo=True, df_speciesAbun_ratio_nonoise=None, num_brep=3, \
                  metabs_cluster_id=None, get_prod=False, B_alone=None, \
                  df_speciesMetab_prod=None, prod_use_prev=True, \
                  use_dilution_term=False, dilution_factor=15000, \
                  use_avg_for_prod=True, check_ratio_dir=True, \
                  return_raw_data=False, mode_Ri=True, Ri=None, power_=1.0):
    num_species = df_speciesMetab_cluster.shape[0]
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    num_metabs = df_speciesMetab_tmp.shape[1]
    if metabs_cluster_id is not None:
        num_metabs = len(metabs_cluster_id)

    # data for dynamic fit
    pass_keep = remove_passages(pass_rm, num_passages=num_passages, num_brep=num_brep)
    df_speciesAbun_prev_tmp = df_speciesAbun_prev_mdl.iloc[:, pass_keep].copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next_mdl.iloc[:, pass_keep].copy()
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_mdl.iloc[:, pass_keep].copy()

    if df_speciesAbun_ratio_nonoise is not None:
        df_speciesAbun_ratio_tmp_nonoise = \
            df_speciesAbun_ratio_nonoise.iloc[:, pass_keep].copy()
    else:
        df_speciesAbun_ratio_tmp_nonoise = df_speciesAbun_ratio_tmp.copy()

    if use_loo:
        Ri_noMicrocosm_dynamicAll_fit_all = {}
        Ri_noMicrocosm_dynamicAll_fit_avg = {}
    Ri_noMicrocosm_dynamicAll_fit_joint = {}
    A_train_joint = {}
    b_train_joint = {}

    for count_p, p_tmp in enumerate(p_vec_new):
        if verbose:
            print(f'count = {count_p}, p_tmp = {p_tmp}')
        Ri_noMicrocosm_dynamicAll_fit_joint[count_p], \
            A_train_joint[count_p], b_train_joint[count_p] = \
                compute_Ri_no_f(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                        df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                        p_tmp, num_passages, range(num_species), \
                        method=method, alpha=alpha, \
                        df_speciesAbun_ratio_nonoise=df_speciesAbun_ratio_tmp_nonoise, \
                        metabs_cluster_id=metabs_cluster_id, get_prod=get_prod, \
                        B_alone=B_alone, \
                        df_speciesMetab_prod=df_speciesMetab_prod, \
                        prod_use_prev=prod_use_prev, \
                        use_dilution_term=use_dilution_term, \
                        dilution_factor=dilution_factor, \
                        use_avg_for_prod=use_avg_for_prod, \
                        check_ratio_dir=check_ratio_dir, mode_Ri=mode_Ri, Ri=Ri, \
                        power_=power_)

        if use_loo:
            if mode_Ri:
                Ri_noMicrocosm_dynamicAll_fit_all[count_p] = \
                    np.zeros((num_species, num_metabs))
                Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = \
                    np.zeros((num_metabs))
            else:
                Ri_noMicrocosm_dynamicAll_fit_all[count_p] = \
                    np.zeros((num_species, num_species))
                Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = \
                    np.zeros((num_species))
        # if use_dilution_term:
        #     Ri_noMicrocosm_dynamicAll_fit_all[count_p] = \
        #         np.zeros((num_species, num_metabs + 1))
        #     Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = \
        #         np.zeros((num_metabs + 1))
            count_species = 0
            for species_ in range(num_species):
                id_species = list(set(range(num_species)) - set([species_]))
                Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :], _, _ = \
                    compute_Ri_no_f(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                            df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                            p_tmp, num_passages, id_species, \
                            method=method, alpha=alpha, \
                            df_speciesAbun_ratio_nonoise=df_speciesAbun_ratio_tmp_nonoise, \
                            metabs_cluster_id=metabs_cluster_id, get_prod=get_prod, \
                            B_alone=B_alone, \
                            df_speciesMetab_prod=df_speciesMetab_prod, \
                            prod_use_prev=prod_use_prev, \
                            use_dilution_term=use_dilution_term, \
                            dilution_factor=dilution_factor, \
                            check_ratio_dir=check_ratio_dir, mode_Ri=mode_Ri, Ri=Ri, \
                            power_=power_)
                if np.sum(Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]) <= 1.5:
                    Ri_noMicrocosm_dynamicAll_fit_avg[count_p] += \
                        Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]
                    count_species += 1




            Ri_noMicrocosm_dynamicAll_fit_avg[count_p] /= count_species


        if use_loo:
            save_obj = {"Ri_noMicrocosm_dynamicAll_fit_all" : \
                        Ri_noMicrocosm_dynamicAll_fit_all, \
                        "Ri_noMicrocosm_dynamicAll_fit_avg" : \
                        Ri_noMicrocosm_dynamicAll_fit_avg, \
                        "Ri_noMicrocosm_dynamicAll_fit_joint" : \
                        Ri_noMicrocosm_dynamicAll_fit_joint}
        else:
            save_obj = {"Ri_noMicrocosm_dynamicAll_fit_joint" : \
                        Ri_noMicrocosm_dynamicAll_fit_joint}
        if save_data:
            with open(file_save, "wb") as file_:
                pickle.dump(save_obj, file_) 
    if not return_raw_data:
        return save_obj
    else:
        return save_obj, A_train_joint, b_train_joint

def fit_dynamic_Ri(df_speciesMetab_cluster, \
                  df_speciesAbun_prev_mdl, df_speciesAbun_next_mdl, \
                  df_speciesAbun_ratio_mdl, p_vec_new, \
                  file_save, num_passages=5, pass_rm=[0, 1, 2], \
                  save_data=True, verbose=True, method="linear", alpha=0, \
                  use_loo=True, df_speciesAbun_ratio_nonoise=None, num_brep=3, \
                  metabs_cluster_id=None, get_prod=False, B_alone=None, \
                  df_speciesMetab_prod=None, prod_use_prev=True, \
                  use_dilution_term=False, dilution_factor=15000, \
                  use_avg_for_prod=True, check_ratio_dir=True, \
                  return_raw_data=False, mode_Ri=True, Ri=None, power_=1.0):
    num_species = df_speciesMetab_cluster.shape[0]
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    num_metabs = df_speciesMetab_tmp.shape[1]
    if metabs_cluster_id is not None:
        num_metabs = len(metabs_cluster_id)

    # data for dynamic fit
    pass_keep = remove_passages(pass_rm, num_passages=num_passages, num_brep=num_brep)
    df_speciesAbun_prev_tmp = df_speciesAbun_prev_mdl.iloc[:, pass_keep].copy()
    df_speciesAbun_next_tmp = df_speciesAbun_next_mdl.iloc[:, pass_keep].copy()
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_mdl.iloc[:, pass_keep].copy()

    if df_speciesAbun_ratio_nonoise is not None:
        df_speciesAbun_ratio_tmp_nonoise = \
            df_speciesAbun_ratio_nonoise.iloc[:, pass_keep].copy()
    else:
        df_speciesAbun_ratio_tmp_nonoise = df_speciesAbun_ratio_tmp.copy()

    if use_loo:
        Ri_noMicrocosm_dynamicAll_fit_all = {}
        Ri_noMicrocosm_dynamicAll_fit_avg = {}
    Ri_noMicrocosm_dynamicAll_fit_joint = {}
    A_train_joint = {}
    b_train_joint = {}

    for count_p, p_tmp in enumerate(p_vec_new):
        if verbose:
            print(f'count = {count_p}, p_tmp = {p_tmp}')
        Ri_noMicrocosm_dynamicAll_fit_joint[count_p], \
            A_train_joint[count_p], b_train_joint[count_p] = \
                compute_Ri(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                        df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                        p_tmp, num_passages, range(num_species), \
                        method=method, alpha=alpha, \
                        df_speciesAbun_ratio_nonoise=df_speciesAbun_ratio_tmp_nonoise, \
                        metabs_cluster_id=metabs_cluster_id, get_prod=get_prod, \
                        B_alone=B_alone, \
                        df_speciesMetab_prod=df_speciesMetab_prod, \
                        prod_use_prev=prod_use_prev, \
                        use_dilution_term=use_dilution_term, \
                        dilution_factor=dilution_factor, \
                        use_avg_for_prod=use_avg_for_prod, \
                        check_ratio_dir=check_ratio_dir, mode_Ri=mode_Ri, Ri=Ri, \
                        power_=power_)

        if use_loo:
            if mode_Ri:
                Ri_noMicrocosm_dynamicAll_fit_all[count_p] = \
                    np.zeros((num_species, num_metabs))
                Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = \
                    np.zeros((num_metabs))
            else:
                Ri_noMicrocosm_dynamicAll_fit_all[count_p] = \
                    np.zeros((num_species, num_species))
                Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = \
                    np.zeros((num_species))
        # if use_dilution_term:
        #     Ri_noMicrocosm_dynamicAll_fit_all[count_p] = \
        #         np.zeros((num_species, num_metabs + 1))
        #     Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = \
        #         np.zeros((num_metabs + 1))
            count_species = 0
            for species_ in range(num_species):
                id_species = list(set(range(num_species)) - set([species_]))
                Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :], _, _ = \
                    compute_Ri(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                            df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                            p_tmp, num_passages, id_species, \
                            method=method, alpha=alpha, \
                            df_speciesAbun_ratio_nonoise=df_speciesAbun_ratio_tmp_nonoise, \
                            metabs_cluster_id=metabs_cluster_id, get_prod=get_prod, \
                            B_alone=B_alone, \
                            df_speciesMetab_prod=df_speciesMetab_prod, \
                            prod_use_prev=prod_use_prev, \
                            use_dilution_term=use_dilution_term, \
                            dilution_factor=dilution_factor, \
                            check_ratio_dir=check_ratio_dir, mode_Ri=mode_Ri, Ri=Ri, \
                            power_=power_)
                if np.sum(Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]) <= 1.5:
                    Ri_noMicrocosm_dynamicAll_fit_avg[count_p] += \
                        Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]
                    count_species += 1




            Ri_noMicrocosm_dynamicAll_fit_avg[count_p] /= count_species


        if use_loo:
            save_obj = {"Ri_noMicrocosm_dynamicAll_fit_all" : \
                        Ri_noMicrocosm_dynamicAll_fit_all, \
                        "Ri_noMicrocosm_dynamicAll_fit_avg" : \
                        Ri_noMicrocosm_dynamicAll_fit_avg, \
                        "Ri_noMicrocosm_dynamicAll_fit_joint" : \
                        Ri_noMicrocosm_dynamicAll_fit_joint}
        else:
            save_obj = {"Ri_noMicrocosm_dynamicAll_fit_joint" : \
                        Ri_noMicrocosm_dynamicAll_fit_joint}
        if save_data:
            with open(file_save, "wb") as file_:
                pickle.dump(save_obj, file_) 
    if not return_raw_data:
        return save_obj
    else:
        return save_obj, A_train_joint, b_train_joint

def fit_dynamic_Ri_with_sim_inoc(df_speciesMetab_cluster, \
                                 df_speciesAbun_prev_mdl, df_speciesAbun_next_mdl, \
                                 df_speciesAbun_ratio_mdl, p_vec_new, \
                                 file_save, num_passages=5, pass_rm=[0, 1, 2], \
                                 use_true_inoc=False, df_speciesAbun_inoc=None, \
                                 use_only_inoc_to_one=False):
    num_species = df_speciesMetab_cluster.shape[0]
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    num_metabs = df_speciesMetab_tmp.shape[1]
    n_breps = 3

    # data for dynamic fit
    pass_keep = remove_passages(pass_rm, num_passages=5)
    df_speciesAbun_prev_tmp = df_speciesAbun_prev_mdl.copy().iloc[:, pass_keep]
    df_speciesAbun_next_tmp = df_speciesAbun_next_mdl.copy().iloc[:, pass_keep]
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_mdl.copy().iloc[:, pass_keep]

    if use_true_inoc:
        vals_inoc_rand = np.zeros((num_species, n_breps))
        for rep_ in range(n_breps):
            vals_inoc_rand[:, rep_] = df_speciesAbun_inoc.iloc[:, 0].values
    else:
        num_rand = 1
        vals_inoc_rand = np.zeros((num_species, n_breps))
        for rep_ in range(n_breps):
            for rand_ in range(num_rand):
                vals_ = np.random.exponential(1 / num_species, num_species)
                vals_ /= np.sum(vals_)
                vals_inoc_rand[:, rep_] += vals_
        vals_inoc_rand /= num_rand
    print(vals_inoc_rand[:, 0])

    sample_names = df_speciesAbun_prev_tmp.columns.values
    num_passes_keep = int(len(pass_keep) / n_breps)
    samples_first = sample_names[0 + np.arange(n_breps) * num_passes_keep]
    inoc_samples = []
    samples_rearrange = []
    for rep_ in range(n_breps):
        inoc_samples.append(f'inoc_r{rep_}')
        # vals_ = 10**np.random.normal(np.log10(1 / num_species), 0.2, num_species)
        # vals_ = np.random.exponential(1 / num_species, num_species)
        # vals_ /= np.sum(vals_)
        vals_ = vals_inoc_rand[:, rep_]
        df_speciesAbun_prev_tmp[inoc_samples[rep_]] = vals_
        df_speciesAbun_next_tmp[inoc_samples[rep_]] = \
            df_speciesAbun_prev_tmp.loc[:, samples_first[rep_]].values
        df_speciesAbun_ratio_tmp[inoc_samples[rep_]] = \
            df_speciesAbun_next_tmp[inoc_samples[rep_]].values / \
                df_speciesAbun_prev_tmp[inoc_samples[rep_]].values
        samples_rearrange.append(f'inoc_r{rep_}')
        for pass_ in range(num_passes_keep):
            samples_rearrange.append(sample_names[rep_ * num_passes_keep + pass_])
    df_speciesAbun_prev_tmp = df_speciesAbun_prev_tmp.loc[:, samples_rearrange]
    df_speciesAbun_next_tmp = df_speciesAbun_next_tmp.loc[:, samples_rearrange]
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_tmp.loc[:, samples_rearrange]
    num_passages += 1

    num_passages_tmp = int(df_speciesAbun_prev_tmp.shape[1] / n_breps)
    if use_only_inoc_to_one:
        id_samples = [0, num_passages_tmp, 2 * num_passages_tmp]
        df_speciesAbun_prev_tmp = df_speciesAbun_prev_tmp.iloc[:, id_samples]
        df_speciesAbun_next_tmp = df_speciesAbun_next_tmp.iloc[:, id_samples]
        df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_tmp.iloc[:, id_samples]
        num_passages = 1

    print(df_speciesAbun_prev_tmp.columns.values)
    Ri_noMicrocosm_dynamicAll_fit_all = {}
    Ri_noMicrocosm_dynamicAll_fit_avg = {}
    Ri_noMicrocosm_dynamicAll_fit_joint = {}

    for count_p, p_tmp in enumerate(p_vec_new):
        print(f'count = {count_p}, p_tmp = {p_tmp}')
        Ri_noMicrocosm_dynamicAll_fit_all[count_p] = np.zeros((num_species, num_metabs))
        Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = np.zeros((num_metabs))
        Ri_noMicrocosm_dynamicAll_fit_joint[count_p] = \
                compute_Ri(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                        df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                        p_tmp, num_passages, range(num_species))

        count_species = 0
        for species_ in range(num_species):
            id_species = list(set(range(num_species)) - set([species_]))
            Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :] = \
                compute_Ri(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                        df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                        p_tmp, num_passages, id_species)
            if np.sum(Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]) <= 1.5:
                Ri_noMicrocosm_dynamicAll_fit_avg[count_p] += \
                    Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]
                count_species += 1


        Ri_noMicrocosm_dynamicAll_fit_avg[count_p] /= count_species


        save_obj = {"Ri_noMicrocosm_dynamicAll_fit_all" : Ri_noMicrocosm_dynamicAll_fit_all, \
                    "Ri_noMicrocosm_dynamicAll_fit_avg" : Ri_noMicrocosm_dynamicAll_fit_avg, \
                    "Ri_noMicrocosm_dynamicAll_fit_joint" : Ri_noMicrocosm_dynamicAll_fit_joint}
        with open(file_save, "wb") as file_:
            pickle.dump(save_obj, file_) 
    
    return save_obj


def fit_dynamic_Ri_with_sim_inoc_bal(df_speciesMetab_cluster, \
                                     df_speciesAbun_prev_mdl, df_speciesAbun_next_mdl, \
                                     df_speciesAbun_ratio_mdl, p_vec_new, \
                                     file_save, num_passages=5, pass_rm=[0, 1, 2]):
    num_species = df_speciesMetab_cluster.shape[0]
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    num_metabs = df_speciesMetab_tmp.shape[1]
    n_breps = 3

    # data for dynamic fit
    pass_keep = remove_passages(pass_rm, num_passages=5)
    df_speciesAbun_prev_tmp = df_speciesAbun_prev_mdl.copy().iloc[:, pass_keep]
    df_speciesAbun_next_tmp = df_speciesAbun_next_mdl.copy().iloc[:, pass_keep]
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_mdl.copy().iloc[:, pass_keep]

    sample_names = df_speciesAbun_prev_tmp.columns.values
    num_passes_keep = int(len(pass_keep) / n_breps)
    samples_first = sample_names[0 + np.arange(n_breps) * num_passes_keep]
    inoc_samples = []
    samples_rearrange = []
    for rep_ in range(n_breps):
        inoc_samples.append(f'inoc_r{rep_}')
        # vals_ = 10**np.random.normal(np.log10(1 / num_species), 0.2, num_species)
        vals_ = np.random.exponential(1 / num_species, num_species)
        vals_ /= np.sum(vals_)
        df_speciesAbun_prev_tmp[inoc_samples[rep_]] = vals_
        df_speciesAbun_next_tmp[inoc_samples[rep_]] = \
            df_speciesAbun_prev_tmp.loc[:, samples_first[rep_]].values
        df_speciesAbun_ratio_tmp[inoc_samples[rep_]] = \
            df_speciesAbun_next_tmp[inoc_samples[rep_]].values / \
                df_speciesAbun_prev_tmp[inoc_samples[rep_]].values
        samples_rearrange.append(f'inoc_r{rep_}')
        for pass_ in range(num_passes_keep):
            samples_rearrange.append(sample_names[rep_ * num_passes_keep + pass_])
    df_speciesAbun_prev_tmp = df_speciesAbun_prev_tmp.loc[:, samples_rearrange]
    df_speciesAbun_next_tmp = df_speciesAbun_next_tmp.loc[:, samples_rearrange]
    df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_tmp.loc[:, samples_rearrange]
    num_passages += 1

    # Ri_noMicrocosm_dynamicAll_fit_all = {}
    # Ri_noMicrocosm_dynamicAll_fit_avg = {}
    Ri_noMicrocosm_dynamicAll_fit_joint = {}

    for count_p, p_tmp in enumerate(p_vec_new):
        print(f'count = {count_p}, p_tmp = {p_tmp}')
        # Ri_noMicrocosm_dynamicAll_fit_all[count_p] = np.zeros((num_species, num_metabs))
        # Ri_noMicrocosm_dynamicAll_fit_avg[count_p] = np.zeros((num_metabs))
        Ri_noMicrocosm_dynamicAll_fit_joint[count_p] = \
                compute_Ri_bal(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                               df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                               p_tmp, num_passages, num_rand=100)
                # compute_Ri(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
                #         df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
                #         p_tmp, num_passages, range(num_species))

        # count_species = 0
        # for species_ in range(num_species):
        #     id_species = list(set(range(num_species)) - set([species_]))
        #     Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :] = \
        #         compute_Ri_bal(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
        #                        df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
        #                        p_tmp, num_passages, num_rand=100)
        #         compute_Ri(df_speciesMetab_tmp.copy(), df_speciesAbun_prev_tmp, \
        #                 df_speciesAbun_next_tmp, df_speciesAbun_ratio_tmp, \
        #                 p_tmp, num_passages, id_species)
        #     if np.sum(Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]) <= 1.5:
        #         Ri_noMicrocosm_dynamicAll_fit_avg[count_p] += \
        #             Ri_noMicrocosm_dynamicAll_fit_all[count_p][species_, :]
        #         count_species += 1


        # Ri_noMicrocosm_dynamicAll_fit_avg[count_p] /= count_species


        # save_obj = {"Ri_noMicrocosm_dynamicAll_fit_all" : Ri_noMicrocosm_dynamicAll_fit_all, \
        #             "Ri_noMicrocosm_dynamicAll_fit_avg" : Ri_noMicrocosm_dynamicAll_fit_avg, \
        #             "Ri_noMicrocosm_dynamicAll_fit_joint" : Ri_noMicrocosm_dynamicAll_fit_joint}
        save_obj = {"Ri_noMicrocosm_dynamicAll_fit_joint" : Ri_noMicrocosm_dynamicAll_fit_joint}
        with open(file_save, "wb") as file_:
            pickle.dump(save_obj, file_) 
    
    return save_obj

def plot_panel_pred_vs_obs_abundance_blind(dir_save_abun_obj, \
                                           df_speciesAbun_mdl, p_tmp=0.1, num_passages=6):

    fig, axes = plt.subplots(2, int(num_passages / 2), \
                             figsize=(40, 35), sharey="row", sharex="col")
    fig.supylabel('predicted abundance (log scale)', fontsize=60)
    fig.supxlabel('observed abundance (log scale)', fontsize=60)
    for pass_ in range(num_passages):
        ax_row_ = int(pass_ / 3)
        ax_col_ = int(pass_ % 3)
        print(f'row = {ax_row_}, col = {ax_col_}')
        save_dir = \
            os.path.abspath(os.path.join(dir_save_abun_obj, \
                                            'predicted_vs_observed_abundance_LeaveOneOutRi', \
                                            'predicted_abundance', \
                                            f'passage_{pass_ + 1}'))
        
        file_save = os.path.abspath(os.path.join(save_dir, 
                                                    f'predicted_abundance_Ri_fit' + \
                                                    f'_with_p{p_tmp}.pickle'))

        with open(file_save, "rb") as file_:
            save_obj = pickle.load(file_)
        growth_rate_all = save_obj['growth_rate_all']
        df_speciesAbun_prev_tmp_ = save_obj['df_speciesAbun_prev']

        df_tmp = pd.DataFrame()
        num_iter = 100
        for col_ in df_speciesAbun_prev_tmp_.columns.values:
            df_tmp[col_] = \
                growth_rate_all[num_iter - 1].copy()[col_].values * \
                df_speciesAbun_prev_tmp_[col_].values
            df_tmp[col_] /= np.sum(df_tmp[col_].values)

        b_ = range(3)
        x = \
            np.array(df_speciesAbun_mdl.copy().iloc[:, [pass_, \
                                                        pass_ + num_passages, \
                                                        pass_ + 2 * num_passages]])
        x = 10**(np.mean(np.log10(x), axis=1)).flatten()
        y = np.array(df_tmp.copy())[:, :].flatten()
        thresh_zero = 1e-8
        y[y == 0] = thresh_zero
        id_ = np.where((x > 0) & (y > 0))[0]
        id_ord = np.where((x > 0) & (y > 0))[0]
        x = np.log10(x[id_])
        y = np.log10(y[id_])
        y[y <= -8] = -8

        plt_ = sns.scatterplot(x=x, \
                                y=y, s=100, ax=axes[ax_row_, ax_col_])

        plt_ = sns.kdeplot(x=x, y=y, fill=True, alpha=0.6, cmap="Reds", \
                        ax=axes[ax_row_, ax_col_])
        plt_.plot([-8, 0], [-8, 0], c="red", linewidth=3)

        corr_val_pe = scipy.stats.pearsonr(10**x, 10**y)
        corr_val_pe_log = scipy.stats.pearsonr(x, y)
        corr_val_sp = scipy.stats.spearmanr(x, y)
        # abs_mean_error = np.median(np.abs(y - x))
        abs_mean_error = np.sqrt(np.mean(np.power(y - x, 2)))
        std_error = np.sqrt(np.std(np.power(y - x, 2)))

        id_notzero_zero = np.where((x > -8) & (y <= -8))[0]

        frac_zero = len(id_notzero_zero) / len(x)

        model = sm.OLS(10**y, 10**x).fit()
        slope = model.params[0]
        slope_pval = model.pvalues[0]
        
        model_log = sm.OLS(y, x).fit()
        slope_log = model_log.params[0]
        slope_log_pval = model_log.pvalues[0]

        # plt_.plot([-8, 0], [np.log10(slope * (1e-8)), np.log10(slope * 1)], c="green")
        plt_.plot([-8, 0], [(slope_log * (-8)), (slope_log * 0)], c="green", \
                    linewidth=3)

        # title_ = f'pearson cc (linear) = {np.round(corr_val_pe[0], 3)}, pval = ' + \
        #                 '{:.3e}'.format(corr_val_pe[1]) + \
        #         f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
        #                 '{:.3e}'.format(corr_val_pe_log[1]) + \
        #         f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
        #                 '{:.3e}'.format(corr_val_sp[1]) + \
        #         f'\n slope = {np.round(slope, 3)} pvalue = ' + \
        #                 '{:.3e}'.format(slope_pval) + \
        #         f', slope_log = {np.round(slope_log, 3)} pvalue = ' + \
        #                 '{:.3e}'.format(slope_log_pval) + \
        #         f'\n abs_median_error = {np.round(abs_mean_error, 3)}' + \
        #         f'\n fit with p = {p_tmp}'
        title_ = f'predicted vs observed abundance at passage {pass_ + 1}' + \
                f'\n pearson cc (log) = {np.round(corr_val_pe_log[0], 3)}, pval = ' + \
                        '{:.3e}'.format(corr_val_pe_log[1]) + \
                f'\n spearman cc = {np.round(corr_val_sp[0], 3)}, pval = ' + \
                        '{:.3e}'.format(corr_val_sp[1]) + \
                f'\n slope_log = {np.round(slope_log, 3)} pvalue = ' + \
                        '{:.3e}'.format(slope_log_pval) + \
                f'\n RMSE = {np.round(abs_mean_error, 3)}'

        axes[ax_row_, ax_col_].set_title(title_, size=30)
        if ax_row_ == 1:
            plt_.set_xticklabels(plt_.get_xticklabels(), fontsize=40)
        if ax_col_ == 0:
            plt_.set_yticklabels(plt_.get_yticklabels(), fontsize=40)

    save_dir = \
        os.path.abspath(os.path.join(dir_save_abun_obj, \
                                        f'predicted_vs_observed_abundance_p{p_tmp}'))
    if not os.path.exists(save_dir):
        # Create a new directory because it does not exist
        os.makedirs(save_dir)
    
    file_save = os.path.abspath(os.path.join(save_dir, 
                                                f'predicted_vs_observed_abundance_' + \
                                                f'consolidated' + \
                                                f'_with_p{p_tmp}.png'))

    fig.figure.savefig(file_save, \
                        dpi=300, transparent=False, facecolor="white")
    plt.close(fig.figure)


# def lasso_drop_metab_exp(df_speciesAbun_mdl, df_speciesAbun_prev, \
#                          df_speciesAbun_next, df_speciesAbun_ratio, p_tmp, \
#                          df_speciesMetab_cluster):
    
def load_data(thresh_zero=1e-8):
    data_dir = os.path.abspath(os.path.join(os.getcwd(), \
                                            '..', 'data', 'jin_pollard'))
    
    # metab names
    # metabolite names and indices of the selected metabolites (metabolites used for the 
    # species-metabolite consumption matrix, c_species^metabolite)
    file_metabNames = os.path.join(data_dir, 
                                    "metabolites_list_FC.csv")
    df_metabNames = pd.read_csv(file_metabNames, sep=",", header=0)

    file_metabIds = os.path.join(data_dir, 
                                    "metabolite_indices.csv")
    df_metabIds = pd.read_csv(file_metabIds, sep=",", header=None)

    metab_names = df_metabNames.iloc[:, 1].values[df_metabIds.iloc[:, 0].values - 1]

    unique_, counts_ = np.unique(metab_names, return_counts=True)

    id_rep = np.where(counts_ > 1)[0]

    for rep_ in id_rep:
        id_m = np.where(metab_names == unique_[rep_])[0]
        
        for count_, id_ in enumerate(id_m):
            metab_names[id_] = f'{metab_names[id_]}_{count_}'
    
    # species names
    # species names and indices of the selected species (metabolites used for the 
    # species abundance matrix, B_species_k)
    file_speciesNames = os.path.join(data_dir, 
                                    "species_list.csv")
    df_speciesNames = pd.read_csv(file_speciesNames, sep=",", header=0)

    file_speciesIds = os.path.join(data_dir, 
                                    "species_indices.csv")
    df_speciesIds = pd.read_csv(file_speciesIds, sep=",", header=None)
    species_names = \
        df_speciesNames.iloc[:, 0].values[df_speciesIds.iloc[:, 0].values - 1]
    

    # species metabolite consumption matrix
    # species-metabolite consumption matrix, c_species^metabolite
    file_speciesMetab = os.path.join(data_dir, 
                                    "species_metabolite_consumption_matrix.csv")
    thresh_zero_metab = 0
    df_speciesMetab = pd.read_csv(file_speciesMetab, sep=",", header=None)
    df_speciesMetab[df_speciesMetab == 0] = thresh_zero_metab
    num_species, num_metabs = df_speciesMetab.shape
    df_speciesMetab.columns = metab_names
    df_speciesMetab.index = species_names

    file_speciesMetab = os.path.join(data_dir, 
                                    "species_metabolite_production_matrix.csv")
    thresh_zero_metab = 0
    df_speciesMetab_prod = pd.read_csv(file_speciesMetab, sep=",", header=None)
    df_speciesMetab_prod[df_speciesMetab_prod == 0] = thresh_zero_metab
    # num_species, num_metabs = df_speciesMetab_prod.shape
    df_speciesMetab_prod.columns = metab_names
    df_speciesMetab_prod.index = species_names

    # species abundance matrix nomicrocosm
    # species abundances for all passages and bioreplicates
    num_passages = 6
    num_bioRep = 3
    file_speciesAbun = os.path.join(data_dir, 
                                    "species_abundances.csv")
    df_speciesAbun = pd.read_csv(file_speciesAbun, sep=",", header=None)
    df_speciesAbun.columns = create_abundance_header_new(num_bioRep=num_bioRep, num_passages=num_passages)
    df_speciesAbun_raw = df_speciesAbun.copy()
    # thresh_zero = 1e-8
    df_speciesAbun[df_speciesAbun == 0] = thresh_zero
    df_speciesAbun.index = species_names
    df_speciesAbun_T = df_speciesAbun.copy()
    df_speciesAbun_T = df_speciesAbun_T.transpose()

    # species abundance matrix agar
    # species abundances for all passages and bioreplicates
    num_passages = 6
    num_bioRep = 3
    file_speciesAbun_super_agar = os.path.join(data_dir, 
                                    "species_abundances_plain_agar_supernatant.csv")
    df_speciesAbun_super_agar = pd.read_csv(file_speciesAbun_super_agar, sep=",", header=None)
    df_speciesAbun_super_agar.columns = \
        create_abundance_header_new(num_bioRep=num_bioRep, num_passages=num_passages)
    df_speciesAbun_super_agar_raw = df_speciesAbun_super_agar.copy()
    # thresh_zero = 1e-8
    df_speciesAbun_super_agar[df_speciesAbun_super_agar == 0] = thresh_zero
    df_speciesAbun_super_agar.index = species_names
    df_speciesAbun_super_agar_T = df_speciesAbun_super_agar.copy()
    df_speciesAbun_super_agar_T = df_speciesAbun_super_agar_T.transpose()

    # species abundance matrix mucin
    # species abundances for all passages and bioreplicates
    num_passages = 6
    num_bioRep = 3
    file_speciesAbun_mucin = os.path.join(data_dir, 
                                    "species_abundances_mucin_supernatant.csv")
    df_speciesAbun_mucin = pd.read_csv(file_speciesAbun_mucin, sep=",", header=None)
    df_speciesAbun_mucin.columns = \
        create_abundance_header_new(num_bioRep=num_bioRep, num_passages=num_passages)
    df_speciesAbun_mucin_raw = df_speciesAbun_mucin.copy()
    # thresh_zero = 1e-8
    df_speciesAbun_mucin[df_speciesAbun_mucin == 0] = thresh_zero
    df_speciesAbun_mucin.index = species_names
    df_speciesAbun_mucin_T = df_speciesAbun_mucin.copy()
    df_speciesAbun_mucin_T = df_speciesAbun_mucin_T.transpose()

    # species abundance matrix inoculum
    file_speciesAbun = os.path.join(data_dir, 
                                    "species_abundances_inoculum.csv")
    df_speciesAbun_inoc = pd.read_csv(file_speciesAbun, sep=",", header=None)
    df_speciesAbun_inoc.columns = ["inoculum"]
    df_speciesAbun_inoc[df_speciesAbun_inoc == 0] = thresh_zero
    df_speciesAbun_inoc.index = species_names

    # species previous, next abundances and growth ratio, no microcosm
    passages_ = np.array(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'])
    reps_ = np.array(['r0', 'r1', 'r2'])
    df_speciesAbun_ratio = pd.DataFrame()
    df_speciesAbun_ratio_corr = pd.DataFrame()
    df_speciesAbun_new = pd.DataFrame()
    df_speciesAbun_prev = pd.DataFrame()
    df_speciesAbun_next = pd.DataFrame()
    df_speciesAbun_split = pd.DataFrame()
    count_ = 0
    brep_vec = list(range(num_bioRep))
    for rep_ in reps_:
        for pass_ in range(1, len(passages_)):
            col_1 = passages_[pass_ - 1] + "_" + rep_
            col_2 = passages_[pass_] + "_" + rep_
            
            df_speciesAbun_prev[col_2] = df_speciesAbun[col_1].values
            df_speciesAbun_next[col_2] = df_speciesAbun[col_2].values
            
            array_1 = np.array(df_speciesAbun[col_2].values / df_speciesAbun[col_1].values)
            array_2 = np.array(df_speciesAbun[col_2].values)
            array_3 = np.array(df_speciesAbun[col_1].values)
            
            df_speciesAbun_new[col_1] = df_speciesAbun[col_1].values
            df_speciesAbun_new[col_2] = df_speciesAbun[col_2].values
            df_speciesAbun_ratio[col_2] = array_1.copy()
            df_speciesAbun_ratio_corr[col_2] = array_1.copy()
            id_ = np.where((array_3 == thresh_zero) & (array_2 == thresh_zero))[0]
    #         id_ = np.where((array_2 == thresh_zero))[0]
    #         id_ = np.where(array_1 == 1)[0]
            
            if len(id_) > 0:
                df_speciesAbun_ratio.iloc[id_, count_] = -1
                rep_alt = list(set(reps_) - set([rep_]))
                ratio_vec = []
                for id_tmp in id_:
                    abun_alt_ratio = []
                    for rep__ in rep_alt:
                        col_1_ = passages_[pass_ - 1] + "_" + rep__
                        col_2_ = passages_[pass_] + "_" + rep__
                        val_1 = df_speciesAbun[col_1_].values[id_tmp]
                        val_2 = df_speciesAbun[col_2_].values[id_tmp]
                        if (val_1 != thresh_zero) | (val_2 != thresh_zero):
                            abun_alt_ratio.append(val_2 / val_1)
                    if len(abun_alt_ratio) != 0:
                        df_speciesAbun_ratio_corr.iloc[id_tmp, count_] = \
                            np.exp(np.mean(np.log10(np.array(abun_alt_ratio))))
                    else:
                        df_speciesAbun_ratio_corr.iloc[id_tmp, count_] = -1
                    
            count_ += 1

    # species previous, next abundances and growth ratio, agar
    # previous and next abundances for supernatant agar
    passages_ = np.array(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'])
    reps_ = np.array(['r0', 'r1', 'r2'])
    df_speciesAbun_super_agar_ratio = pd.DataFrame()
    df_speciesAbun_super_agar_ratio_corr = pd.DataFrame()
    df_speciesAbun_super_agar_new = pd.DataFrame()
    df_speciesAbun_super_agar_prev = pd.DataFrame()
    df_speciesAbun_super_agar_next = pd.DataFrame()
    df_speciesAbun_super_agar_split = pd.DataFrame()
    count_ = 0
    brep_vec = list(range(num_bioRep))
    for rep_ in reps_:
        for pass_ in range(1, len(passages_)):
            col_1 = passages_[pass_ - 1] + "_" + rep_
            col_2 = passages_[pass_] + "_" + rep_
            
            df_speciesAbun_super_agar_prev[col_2] = df_speciesAbun_super_agar[col_1].values
            df_speciesAbun_super_agar_next[col_2] = df_speciesAbun_super_agar[col_2].values
            
            array_1 = np.array(df_speciesAbun_super_agar[col_2].values / df_speciesAbun_super_agar[col_1].values)
            array_2 = np.array(df_speciesAbun_super_agar[col_2].values)
            array_3 = np.array(df_speciesAbun_super_agar[col_1].values)
            
            df_speciesAbun_super_agar_new[col_1] = df_speciesAbun_super_agar[col_1].values
            df_speciesAbun_super_agar_new[col_2] = df_speciesAbun_super_agar[col_2].values
            df_speciesAbun_super_agar_ratio[col_2] = array_1.copy()
            df_speciesAbun_super_agar_ratio_corr[col_2] = array_1.copy()
            id_ = np.where((array_3 == thresh_zero) & (array_2 == thresh_zero))[0]
    #         id_ = np.where((array_2 == thresh_zero))[0]
    #         id_ = np.where(array_1 == 1)[0]
            
            if len(id_) > 0:
                df_speciesAbun_super_agar_ratio.iloc[id_, count_] = -1
                rep_alt = list(set(reps_) - set([rep_]))
                ratio_vec = []
                for id_tmp in id_:
                    Abun_super_agar_alt_ratio = []
                    for rep__ in rep_alt:
                        col_1_ = passages_[pass_ - 1] + "_" + rep__
                        col_2_ = passages_[pass_] + "_" + rep__
                        val_1 = df_speciesAbun_super_agar[col_1_].values[id_tmp]
                        val_2 = df_speciesAbun_super_agar[col_2_].values[id_tmp]
                        if (val_1 != thresh_zero) | (val_2 != thresh_zero):
                            Abun_super_agar_alt_ratio.append(val_2 / val_1)
                    if len(Abun_super_agar_alt_ratio) != 0:
                        df_speciesAbun_super_agar_ratio_corr.iloc[id_tmp, count_] = \
                            np.exp(np.mean(np.log10(np.array(Abun_super_agar_alt_ratio))))
                    else:
                        df_speciesAbun_super_agar_ratio_corr.iloc[id_tmp, count_] = -1
                    
            count_ += 1

    # species previous, next abundances and growth ratio, mucin
    # previous and next abundances for mucin agar
    passages_ = np.array(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'])
    reps_ = np.array(['r0', 'r1', 'r2'])
    df_speciesAbun_mucin_ratio = pd.DataFrame()
    df_speciesAbun_mucin_ratio_corr = pd.DataFrame()
    df_speciesAbun_mucin_new = pd.DataFrame()
    df_speciesAbun_mucin_prev = pd.DataFrame()
    df_speciesAbun_mucin_next = pd.DataFrame()
    df_speciesAbun_mucin_split = pd.DataFrame()
    count_ = 0
    brep_vec = list(range(num_bioRep))
    for rep_ in reps_:
        for pass_ in range(1, len(passages_)):
            col_1 = passages_[pass_ - 1] + "_" + rep_
            col_2 = passages_[pass_] + "_" + rep_
            
            df_speciesAbun_mucin_prev[col_2] = df_speciesAbun_mucin[col_1].values
            df_speciesAbun_mucin_next[col_2] = df_speciesAbun_mucin[col_2].values
            
            array_1 = np.array(df_speciesAbun_mucin[col_2].values / df_speciesAbun_mucin[col_1].values)
            array_2 = np.array(df_speciesAbun_mucin[col_2].values)
            array_3 = np.array(df_speciesAbun_mucin[col_1].values)
            
            df_speciesAbun_mucin_new[col_1] = df_speciesAbun_mucin[col_1].values
            df_speciesAbun_mucin_new[col_2] = df_speciesAbun_mucin[col_2].values
            df_speciesAbun_mucin_ratio[col_2] = array_1.copy()
            df_speciesAbun_mucin_ratio_corr[col_2] = array_1.copy()
            id_ = np.where((array_3 == thresh_zero) & (array_2 == thresh_zero))[0]
    #         id_ = np.where((array_2 == thresh_zero))[0]
    #         id_ = np.where(array_1 == 1)[0]
            
            if len(id_) > 0:
                df_speciesAbun_mucin_ratio.iloc[id_, count_] = -1
                rep_alt = list(set(reps_) - set([rep_]))
                ratio_vec = []
                for id_tmp in id_:
                    Abun_mucin_alt_ratio = []
                    for rep__ in rep_alt:
                        col_1_ = passages_[pass_ - 1] + "_" + rep__
                        col_2_ = passages_[pass_] + "_" + rep__
                        val_1 = df_speciesAbun_mucin[col_1_].values[id_tmp]
                        val_2 = df_speciesAbun_mucin[col_2_].values[id_tmp]
                        if (val_1 != thresh_zero) | (val_2 != thresh_zero):
                            Abun_mucin_alt_ratio.append(val_2 / val_1)
                    if len(Abun_mucin_alt_ratio) != 0:
                        df_speciesAbun_mucin_ratio_corr.iloc[id_tmp, count_] = \
                            np.exp(np.mean(np.log10(np.array(Abun_mucin_alt_ratio))))
                    else:
                        df_speciesAbun_mucin_ratio_corr.iloc[id_tmp, count_] = -1
                    
            count_ += 1

    return df_speciesMetab, df_speciesAbun, \
            df_speciesAbun_super_agar, \
            df_speciesAbun_mucin, \
            df_speciesAbun_inoc, \
            df_speciesAbun_ratio, \
            df_speciesAbun_super_agar_ratio, \
            df_speciesAbun_mucin_ratio, \
            df_speciesAbun_prev, \
            df_speciesAbun_super_agar_prev, \
            df_speciesAbun_mucin_prev, \
            df_speciesAbun_next, \
            df_speciesAbun_super_agar_next, \
            df_speciesAbun_mucin_next, metab_names, \
            species_names, df_speciesMetab_prod

def load_data_manuscript(thresh_zero=1e-8, data_dir=None):
    """
    Load data for the figures in the manuscript.

    Parameters:
    - thresh_zero (float): Threshold value for replacing zeros in the strain abundance data. Default is 1e-8.
    - data_dir (str): Directory path to the data. If None, the default data directory is used.

    Returns:
    - df_speciesMetab (pandas.DataFrame): Species-metabolite consumption matrix, c_{\\alpha, i}.
    - df_speciesAbun (pandas.DataFrame): Species abundance matrix, N_{alpha}(end of cycle k).
    - df_speciesAbun_inoc (pandas.DataFrame): Species abundance matrix for the inoculum, N_{alpha}(inoculum).
    - df_speciesAbun_ratio (pandas.DataFrame): Species abundance ratio matrix, N_{alpha}(end of cycle k) / N_{alpha}(end of cycle k - 1).
    - df_speciesAbun_prev (pandas.DataFrame): Species abundance matrix for previous passages N_{alpha}(end of cycle k - 1).
    - df_speciesAbun_next (pandas.DataFrame): Species abundance matrix for next passages, N_{alpha}(end of cycle k).
    - metab_names (numpy.ndarray): Array of metabolite names.
    - species_names (numpy.ndarray): Array of species names.
    - df_speciesMetab_prod (pandas.DataFrame): Species-metabolite production matrix, p_{\\gamma, i}.
    """
    # path to the data
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.getcwd(), \
                                                '..', 'data', 'jin_pollard'))
    
    # metab names
    # metabolite names and indices of the selected metabolites (metabolites used for the 
    # species-metabolite consumption matrix, c_species^metabolite)
    # metabolites_list_FC.csv contains the names of the metabolites
    file_metabNames = os.path.join(data_dir, 
                                    "metabolites_list_FC.csv")
    df_metabNames = pd.read_csv(file_metabNames, sep=",", header=0)

    # metabolite_indices.csv contains the indices of the metabolites used in the
    # species-metabolite consumption matrix, c_species^metabolite for the 63 species and 
    # 292 metabolites
    file_metabIds = os.path.join(data_dir, 
                                    "metabolite_indices.csv")
    df_metabIds = pd.read_csv(file_metabIds, sep=",", header=None)

    # metabolite names used in the species-metabolite consumption matrix, c_species^metabolite
    metab_names = df_metabNames.iloc[:, 1].values[df_metabIds.iloc[:, 0].values - 1]
    unique_, counts_ = np.unique(metab_names, return_counts=True)
    id_rep = np.where(counts_ > 1)[0] # indices of the repeated metabolites
    # repeated metabolites are renamed to include the index of the repetition
    for rep_ in id_rep:
        id_m = np.where(metab_names == unique_[rep_])[0]
        for count_, id_ in enumerate(id_m):
            metab_names[id_] = f'{metab_names[id_]}_{count_}'
    
    # species names
    # species names and indices of the selected species (metabolites used for the 
    # species abundance matrix, N_alpha)
    # species_list.csv contains the names of the species
    file_speciesNames = os.path.join(data_dir, 
                                    "species_list.csv")
    df_speciesNames = pd.read_csv(file_speciesNames, sep=",", header=0)

    # species_indices.csv contains the indices of the species used in the
    # species abundance matrix, N_alpha, and species-metabolite consumption matrix, 
    # c_species^metabolite for the 63 species and 292 metabolites
    file_speciesIds = os.path.join(data_dir, 
                                    "species_indices.csv")
    df_speciesIds = pd.read_csv(file_speciesIds, sep=",", header=None)
    # species names used in the species abundance matrix, N_alpha, and species-metabolite
    # consumption matrix, c_species^metabolite
    species_names = \
        df_speciesNames.iloc[:, 0].values[df_speciesIds.iloc[:, 0].values - 1]
    
    # species_metabolite_consumption_matrix.csv contains the species-metabolite consumption
    # matrix, c_species^metabolite for the 63 species and 292 metabolites
    file_speciesMetab = os.path.join(data_dir, 
                                    "species_metabolite_consumption_matrix.csv")
    thresh_zero_metab = 0
    df_speciesMetab = pd.read_csv(file_speciesMetab, sep=",", header=None)
    df_speciesMetab[df_speciesMetab == 0] = thresh_zero_metab
    num_species, num_metabs = df_speciesMetab.shape
    df_speciesMetab.columns = metab_names
    df_speciesMetab.index = species_names

    # species_metabolite_production_matrix.csv contains the species-metabolite production
    # matrix, c_species^metabolite for the 63 species and 292 metabolites
    file_speciesMetab = os.path.join(data_dir, 
                                    "species_metabolite_production_matrix.csv")
    thresh_zero_metab = 0
    df_speciesMetab_prod = pd.read_csv(file_speciesMetab, sep=",", header=None)
    df_speciesMetab_prod[df_speciesMetab_prod == 0] = thresh_zero_metab
    # num_species, num_metabs = df_speciesMetab_prod.shape
    df_speciesMetab_prod.columns = metab_names
    df_speciesMetab_prod.index = species_names

    # species abundance matrix for the liquid-media only control from Jin et al. 2023
    # species abundances for all 6 passages and 3 bioreplicates
    num_passages = 6
    num_bioRep = 3
    file_speciesAbun = os.path.join(data_dir, 
                                    "species_abundances.csv")
    df_speciesAbun = pd.read_csv(file_speciesAbun, sep=",", header=None)
    df_speciesAbun.columns = create_abundance_header_new(num_bioRep=num_bioRep, num_passages=num_passages)
    df_speciesAbun_raw = df_speciesAbun.copy()
    # thresh_zero = 1e-8
    df_speciesAbun[df_speciesAbun == 0] = thresh_zero
    df_speciesAbun.index = species_names
    df_speciesAbun_T = df_speciesAbun.copy()
    df_speciesAbun_T = df_speciesAbun_T.transpose()

    # species abundance matrix for the inoculum
    file_speciesAbun = os.path.join(data_dir, 
                                    "species_abundances_inoculum.csv")
    df_speciesAbun_inoc = pd.read_csv(file_speciesAbun, sep=",", header=None)
    df_speciesAbun_inoc.columns = ["inoculum"]
    df_speciesAbun_inoc[df_speciesAbun_inoc == 0] = thresh_zero
    df_speciesAbun_inoc.index = species_names

    # species abundance matrix for previous abundances, next abundances and ratio
    # for the liquid-media only control from Jin et al. 2023
    passages_ = np.array(['p1', 'p2', 'p3', 'p4', 'p5', 'p6'])
    reps_ = np.array(['r0', 'r1', 'r2'])
    df_speciesAbun_ratio = pd.DataFrame()
    df_speciesAbun_ratio_corr = pd.DataFrame()
    df_speciesAbun_new = pd.DataFrame()
    df_speciesAbun_prev = pd.DataFrame()
    df_speciesAbun_next = pd.DataFrame()
    df_speciesAbun_split = pd.DataFrame()
    count_ = 0
    brep_vec = list(range(num_bioRep))
    for rep_ in reps_:
        for pass_ in range(1, len(passages_)):
            col_1 = passages_[pass_ - 1] + "_" + rep_
            col_2 = passages_[pass_] + "_" + rep_
            
            df_speciesAbun_prev[col_2] = df_speciesAbun[col_1].values
            df_speciesAbun_next[col_2] = df_speciesAbun[col_2].values
            
            array_1 = np.array(df_speciesAbun[col_2].values / df_speciesAbun[col_1].values)
            array_2 = np.array(df_speciesAbun[col_2].values)
            array_3 = np.array(df_speciesAbun[col_1].values)
            
            df_speciesAbun_new[col_1] = df_speciesAbun[col_1].values
            df_speciesAbun_new[col_2] = df_speciesAbun[col_2].values
            df_speciesAbun_ratio[col_2] = array_1.copy()
            df_speciesAbun_ratio_corr[col_2] = array_1.copy()
            id_ = np.where((array_3 == thresh_zero) & (array_2 == thresh_zero))[0]
    #         id_ = np.where((array_2 == thresh_zero))[0]
    #         id_ = np.where(array_1 == 1)[0]
            
            if len(id_) > 0:
                df_speciesAbun_ratio.iloc[id_, count_] = -1
                rep_alt = list(set(reps_) - set([rep_]))
                ratio_vec = []
                for id_tmp in id_:
                    abun_alt_ratio = []
                    for rep__ in rep_alt:
                        col_1_ = passages_[pass_ - 1] + "_" + rep__
                        col_2_ = passages_[pass_] + "_" + rep__
                        val_1 = df_speciesAbun[col_1_].values[id_tmp]
                        val_2 = df_speciesAbun[col_2_].values[id_tmp]
                        if (val_1 != thresh_zero) | (val_2 != thresh_zero):
                            abun_alt_ratio.append(val_2 / val_1)
                    if len(abun_alt_ratio) != 0:
                        df_speciesAbun_ratio_corr.iloc[id_tmp, count_] = \
                            np.exp(np.mean(np.log10(np.array(abun_alt_ratio))))
                    else:
                        df_speciesAbun_ratio_corr.iloc[id_tmp, count_] = -1
                    
            count_ += 1

    return df_speciesMetab, df_speciesAbun, \
            df_speciesAbun_inoc, \
            df_speciesAbun_ratio, \
            df_speciesAbun_prev, \
            df_speciesAbun_next, metab_names, \
            species_names, df_speciesMetab_prod

def get_prod_term(df_speciesMetab_prod, df_speciesAbun_split, B_alone, \
                  get_prod=False):
    prod_metabs_cond = {}

    for col_ in df_speciesAbun_split.columns.values:
        if get_prod:
            prod_metabs_cond[col_] = \
                1 + np.matmul(np.array(df_speciesMetab_prod.copy()).transpose(), \
                              df_speciesAbun_split[col_].values / B_alone)
            prod_metabs_cond[col_] = \
                np.matmul(np.ones((df_speciesAbun_split.shape[0], 1)), \
                          prod_metabs_cond[col_].reshape(1, df_speciesMetab_prod.shape[1]))
        else:
            prod_metabs_cond[col_] = \
                np.ones((df_speciesAbun_split.shape[0], df_speciesMetab_prod.shape[1]))
    return prod_metabs_cond

def metabs_to_remove_knockdown_species(df_speciesMetab, species_rm, \
                                       id_species_sensitive=None):
    if id_species_sensitive is None:
        id_species_sensitive = \
            np.arange(df_speciesMetab.shape[0])
    id_metabs = \
        np.where(np.array(df_speciesMetab)[species_rm, :].flatten() != 0)[0]

    id_metabs_rm = []
    for metab_ in id_metabs:
        id_species = \
            np.where(np.array(df_speciesMetab)[id_species_sensitive, metab_] != 0)[0]
        if len(id_species) <= 1:
            id_metabs_rm.append(metab_)

    id_metabs = np.arange((df_speciesMetab.shape[1]))
    id_metabs = np.delete(id_metabs, id_metabs_rm)
    return id_metabs

def get_RMSE_Balone_func(df_speciesMetab_cluster, df_speciesMetab_prod_cluster, p_vec_new, count_p, Ri_all, \
                    df_speciesAbun_inoc, df_speciesAbun_mdl, df_speciesAbun_prev_mdl, \
                    df_speciesAbun_ratio_mdl, abun_alone, \
                    num_passages=6, use_dilution=False, \
                    dilution_factor=15000):
    # evalute Ri with production
    num_metabs_clust = df_speciesMetab_cluster.shape[1]
    df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
    df_speciesMetab_prod_tmp = df_speciesMetab_prod_cluster.copy()
    # count_p = 7

    RMSE_mat_full = np.zeros((num_passages, 2))
    Ri_avg = Ri_all[count_p].copy()
    Ri_fit = {0: Ri_avg}
    # RMSE_sens_complete_full = np.zeros((num_passages - 1, 2))
    sens_obj_all_prod, RMSE_obj_all_prod = \
        blindly_pred_abun_growth([p_vec_new[count_p]], df_speciesMetab_tmp, \
                                 df_speciesAbun_inoc, df_speciesAbun_mdl, \
                                 df_speciesAbun_prev_mdl, \
                                 df_speciesAbun_ratio_mdl, \
                                 Ri_fit, "dummy", "dummy", \
                                 "dummy", num_passages=6, num_iter=100, \
                                 thresh_zero=1e-8, Ri_ss=False, plot_=False, \
                                 save_data_obj=False, \
                                 return_sensitivity_ana=True, \
                                 get_prod=True, B_alone=abun_alone, \
                                 df_speciesMetab_prod=df_speciesMetab_prod_tmp, \
                                 prod_use_prev=False, use_dilution=use_dilution, \
                                 dilution_factor=dilution_factor)
    for pass_ in range(num_passages):
        RMSE_mat_full[pass_, 0] = RMSE_obj_all_prod[pass_]["abundance"]
        RMSE_mat_full[pass_, 1] = RMSE_obj_all_prod[pass_]["growth_ratio"]
    return RMSE_mat_full

def infer_substrate_abundance():
    return 0

def get_B_alone_func(Ri_avg, df_speciesMetab_cluster):
    num_metabs_clust = df_speciesMetab_cluster.shape[1]
    num_species = df_speciesMetab_cluster.shape[0]
    abun_alone = np.zeros((num_species))

    for species_ in range(num_species):
        abun_alone[species_] = np.sum(Ri_avg * np.array(df_speciesMetab_cluster.iloc[species_, :].values))
    # B_alone = np.exp(np.mean(np.log(abun_alone)))
    return abun_alone

def get_RMSE_against_uniform_sd(p_vec_new_tmp, \
                                df_speciesMetab_cluster, \
                                df_speciesMetab_prod_cluster, \
                                B_alone_cur, \
                                df_speciesAbun_prev_sim, \
                                df_speciesAbun_next_sim, \
                                df_speciesAbun_ratio_sim, \
                                n_repeats=100):
    rand_seed = 7363
    np.random.seed(rand_seed)

    Ri_dist = {}

    for s_ in range(n_repeats):
        df_speciesAbun_prev_tmp = df_speciesAbun_prev_sim.copy()
        df_speciesAbun_next_tmp = df_speciesAbun_next_sim.copy()
        df_speciesAbun_ratio_tmp = df_speciesAbun_ratio_sim.copy()

        for pass_ in range(6):
            id_ = np.random.choice(np.arange(3), 3, replace=True)
            id_pass = np.arange(pass_ * 3, (pass_ + 1) * 3)
            # id_pass = id_pass[id_]
            df_speciesAbun_prev_tmp.iloc[:, id_pass] = \
                df_speciesAbun_prev_sim.iloc[:, id_pass[id_]].values
            df_speciesAbun_next_tmp.iloc[:, id_pass] = \
                df_speciesAbun_next_sim.iloc[:, id_pass[id_]].values
            df_speciesAbun_ratio_tmp.iloc[:, id_pass] = \
                df_speciesAbun_ratio_sim.iloc[:, id_pass[id_]].values


        num_metabs_clust = df_speciesMetab_cluster.shape[1]
        df_speciesMetab_tmp = df_speciesMetab_cluster.copy()
        df_speciesMetab_prod_tmp = df_speciesMetab_prod_cluster.copy()
        Ri_dynamic_obj = {}
        # Ri_dynamic_obj = fit_dynamic_Ri(df_speciesMetab_cluster, \
        #                                 df_speciesAbun_prev_mdl, df_speciesAbun_next_mdl, \
        #                                 df_speciesAbun_ratio_mdl, p_vec_new, \
        #                                 file_save, num_passages=5, pass_rm=[1, 2, 3, 4])
        num_passages = 7
        pass_keep = np.arange(num_passages - 1)
        id_keep = list(pass_keep) + list(pass_keep + num_passages - 1) + list(pass_keep + 2 * (num_passages - 1))
        pass_rm = [0, 2, 3, 4, 5]
            
            
        Ri_dynamic_obj, A_train, b_train = fit_dynamic_Ri(df_speciesMetab_tmp, \
                                        df_speciesAbun_prev_tmp.iloc[:, id_keep], \
                                        df_speciesAbun_next_tmp.iloc[:, id_keep], \
                                        df_speciesAbun_ratio_tmp.iloc[:, id_keep], p_vec_new_tmp, \
                                        None, num_passages=6, pass_rm=pass_rm, save_data=False, \
                                        num_brep=3, \
                                        get_prod=True, B_alone=B_alone_cur, \
                                        df_speciesMetab_prod=df_speciesMetab_prod_tmp, \
                                        prod_use_prev=False, use_avg_for_prod=False, \
                                        return_raw_data=True, check_ratio_dir=False)
        Ri_dist[s_] = \
            Ri_dynamic_obj['Ri_noMicrocosm_dynamicAll_fit_joint'][0].copy()
    return Ri_dist



def get_competition(Ri_avg, df_speciesMetab_cluster, id_1, id_2):
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    vec_src = df_speciesMetab_cluster.iloc[id_1, :].values
    vec_dest = df_speciesMetab_cluster.iloc[id_2, :].values
    nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
    nnz_src = len(np.where(Ri_avg * vec_src != 0)[0])
    nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
#     competition_rand[b_] = nnz_common / nnz_dest
#         competition_rand[b_] = tmp
    return nnz_common / nnz_dest
    # return nnz_common / ((nnz_dest + nnz_src) / 2)

def get_competition_weighted(Ri_avg, df_speciesMetab_cluster, id_1, id_2):
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    vec_src = df_speciesMetab_cluster.iloc[id_1, :].values
    vec_dest = df_speciesMetab_cluster.iloc[id_2, :].values
    nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
    nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
#     competition_rand[b_] = nnz_common / nnz_dest
#         competition_rand[b_] = tmp

    x = vec_src
    y = vec_dest
    id_ = np.where(Ri_avg > 0)[0]
    # x = x[id_] * Ri_avg[id_]
    # y = y[id_] * Ri_avg[id_]

    # competition_ = np.sum(x * y) / \
    #     (np.sqrt(np.sum(np.power(x, 2))) * \
    #         np.sqrt(np.sum(np.power(y, 2))))
    num_ = x * y * Ri_avg
    denom_1 = y * Ri_avg
    denom_2 = x * Ri_avg
    Ri_avg_log = np.log10(Ri_avg.copy() + 1e-12)
    id_ = np.where(Ri_avg > 0)[0]
    ri_min = np.min(Ri_avg_log[id_])
    Ri_avg_log += (-ri_min) + 1
    # competition_ = (2 * np.sum(Ri_avg_log[num_ > 0])) / \
    #     (np.sum(Ri_avg_log[denom_1 > 0]) + np.sum(Ri_avg_log[denom_2 > 0]))
    competition_ = (2 * (np.sum((Ri_avg)[num_ > 0]))) / \
        ((np.sum(Ri_avg[denom_1 > 0])) + \
         (np.sum(Ri_avg[denom_2 > 0])))
    return competition_

def get_competition_weighted_2(Ri_avg, df_speciesMetab_cluster, id_1, id_2, abun_1, abun_2, \
                               metab_rm=None):
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    vec_src = df_speciesMetab_cluster.iloc[id_1, :].values
    vec_dest = df_speciesMetab_cluster.iloc[id_2, :].values
    nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
    nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
#     competition_rand[b_] = nnz_common / nnz_dest
#         competition_rand[b_] = tmp
    Ri_tmp = Ri_avg.copy()
    x = vec_src * Ri_tmp.copy()
    y = vec_dest * Ri_tmp.copy()
    if metab_rm is not None:
        Ri_tmp[metab_rm] = 0
    id_ = np.where(Ri_tmp > 0)[0]
    x = x[id_]
    y = y[id_]
    nnz_dest = len(np.where(y != 0)[0])
    nnz_common = len(np.where((x * y) != 0)[0])
    competition_ = nnz_common / nnz_dest

    # competition_ = np.sum(x * y) / \
    #     (np.sqrt(np.sum(np.power(x, 2))) * \
    #         np.sqrt(np.sum(np.power(y, 2))))
    # model = sm.OLS(x, y).fit()
    # slope_ = model.params[0]
    # competition_ = np.arctan(np.array([slope_])) / (np.pi / 2)
    # num_ = x * y * Ri_avg
    # denom_1 = x * Ri_avg
    # denom_2 = y * Ri_avg
    # Ri_avg_log = np.log10(Ri_avg.copy() + 1e-12)
    # id_ = np.where(Ri_avg > 0)[0]
    # ri_min = np.min(Ri_avg_log[id_])
    # Ri_avg_log += (-ri_min) + 1
    # # competition_ = (2 * np.sum(Ri_avg_log[num_ > 0])) / \
    # #     (np.sum(Ri_avg_log[denom_1 > 0]) + np.sum(Ri_avg_log[denom_2 > 0]))
    # competition_ = (2 * (np.sum((x * y * Ri_avg**2)[num_ > 0]))) / \
    #     ((np.sum((x * Ri_avg)[denom_1 > 0]) * abun_2) + \
    #      (np.sum((y * Ri_avg)[denom_2 > 0]) * abun_1))
    # competition_ = np.sum(x >= y) / len(x)
    return competition_

def get_crossfeeding(Ri_avg, df_speciesMetab_cluster, \
                     df_speciesMetab_prod_cluster, id_1, id_2):
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    vec_src = df_speciesMetab_prod_cluster.iloc[int(id_1), :].values
    vec_dest = df_speciesMetab_cluster.iloc[int(id_2), :].values
    nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
    nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
#         crossfeeding_rand[b_] = tmp
    return nnz_common / nnz_dest 

def random_competition_index(Ri_avg, df_speciesMetab_cluster, id_src, nboot=10000):
    competition_rand = np.zeros((nboot))
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    for b_ in range(nboot):
        id_1 = np.random.choice(id_src, 1, replace=False)[0]
        id_species_left = id_species.copy()
        id_species_left = np.delete(id_species_left, [id_1])
        id_2 = np.random.choice(id_species_left, 1, replace=False)[0]
        vec_src = df_speciesMetab_cluster.iloc[int(id_1), :].values
        vec_dest = df_speciesMetab_cluster.iloc[int(id_2), :].values
        nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
        nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
        tmp = np.sum(Ri_avg * vec_src * Ri_avg * vec_dest) / \
            (np.sqrt(np.sum(np.power(Ri_avg * vec_src, 2))) * \
             np.sqrt(np.sum(np.power(Ri_avg * vec_dest, 2))))
        competition_rand[b_] = nnz_common / nnz_dest
#         competition_rand[b_] = tmp
    return competition_rand

def random_competition_subset(Ri_avg, df_speciesMetab_cluster, id_src, id_species_left, len_dest, nboot=10000):
    competition_rand = np.zeros((nboot))
    competition_rand_id = np.zeros((nboot), dtype='long')
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    for b_ in range(nboot):
        id_1 = np.random.choice(id_src, 1, replace=False)[0]
        # id_species_left = id_species.copy()
        # id_species_left = np.delete(id_species_left, [id_1])
        id_dest = np.random.choice(id_species_left, len_dest, replace=False)
        competition_rand_tmp = np.zeros((len_dest))
        for j in range(len_dest):
            id_2 = id_dest[j]
            competition_rand_tmp[j] = get_competition(Ri_avg, df_speciesMetab_cluster, id_2, id_1)
        competition_rand[b_] = np.max(competition_rand_tmp)
        competition_rand_id[b_] = id_dest[np.argmax(competition_rand_tmp)]
#         competition_rand[b_] = tmp
    return competition_rand, competition_rand_id

def random_competition_with_blue_index(Ri_avg, df_speciesMetab_cluster, id_src, perturbed_nodes_nnz, nboot=10000):
    competition_rand = np.zeros((nboot))
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    for b_ in range(nboot):
        id_1 = np.random.choice(id_src, 1, replace=False)[0]
        id_species_left = id_species.copy()
        id_species_left = np.delete(id_species_left, [id_1])
        num_red = np.random.choice(perturbed_nodes_nnz, 1, replace=False)[0]
        id_2_vec = np.random.choice(id_species_left, num_red, replace=False)
        vec_src = df_speciesMetab_cluster.iloc[int(id_1), :].values
        competition_rand_tmp = []
        for id_2 in id_2_vec:
            vec_dest = df_speciesMetab_cluster.iloc[int(id_2), :].values
            nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
            nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
            tmp = np.sum(Ri_avg * vec_src * Ri_avg * vec_dest) / \
                (np.sqrt(np.sum(np.power(Ri_avg * vec_src, 2))) * \
                 np.sqrt(np.sum(np.power(Ri_avg * vec_dest, 2))))
#             competition_rand[b_] = nnz_common / nnz_dest
#             competition_rand_tmp.append(tmp)
            competition_rand_tmp.append(nnz_common / nnz_dest)
        competition_rand[b_] = np.median(np.array(competition_rand_tmp))
    return competition_rand

def random_crossfeeding_index(Ri_avg, df_speciesMetab_cluster, \
                              df_speciesMetab_prod_cluster, id_src, nboot=10000):
    crossfeeding_rand = np.zeros((nboot))
    num_species = df_speciesMetab_cluster.shape[0]
    id_species = np.arange(num_species)
    for b_ in range(nboot):
        id_1 = np.random.choice(id_src, 1, replace=False)[0]
        id_species_left = id_species.copy()
        id_species_left = np.delete(id_species_left, [id_1])
        id_2 = np.random.choice(id_species_left, 1, replace=False)[0]
        vec_src = df_speciesMetab_prod_cluster.iloc[int(id_1), :].values
        vec_dest = df_speciesMetab_cluster.iloc[int(id_2), :].values
        nnz_dest = len(np.where(Ri_avg * vec_dest != 0)[0])
        nnz_common = len(np.where((Ri_avg * vec_dest * vec_src) != 0)[0])
        tmp = np.sum(vec_src * vec_dest) / \
            (np.sqrt(np.sum(np.power(vec_src, 2))) * np.sqrt(np.sum(np.power(vec_dest, 2))))
        crossfeeding_rand[b_] = nnz_common / nnz_dest
#         crossfeeding_rand[b_] = tmp
    return crossfeeding_rand    
