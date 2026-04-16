#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09


"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np
import torch


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # 输入参数说明：
    # probFea: query特征 [num_query, D]
    # galFea: gallery特征 [num_gallery, D]
    # k1: k-互近邻的k值（核心参数，控制近邻数量）
    # k2: 二次近邻融合的k值（控制平滑程度）
    # lambda_value: 初始距离和Jaccard距离的融合系数
    # local_distmat: 局部距离矩阵（可选，融合局部特征）
    # only_local: 是否仅使用局部距离（默认False）

    # 1. 转换为tensor（兼容numpy输入）+ 统计样本数
    query_num = probFea.size(0)  # query样本数
    all_num = query_num + galFea.size(0)  # query+gallery总样本数

    # 2. 计算初始距离矩阵（全局欧式距离）
    if only_local:
        original_dist = local_distmat  # 仅用局部距离（极少用）
    else:
        # 拼接query和gallery特征，统一计算全局距离
        feat = torch.cat([probFea, galFea])  # [all_num, D]

        # 快速计算欧式距离矩阵（矩阵运算优化，避免双重循环）
        # 欧式距离公式：||a-b||² = ||a||² + ||b||² - 2a·b
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
                  torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1, -2, feat, feat.t())  # 等价于 distmat = distmat - 2*feat@feat.T

        original_dist = distmat.cpu().numpy()  # 转到CPU，转为numpy
        del feat  # 释放显存
        if not local_distmat is None:
            original_dist = original_dist + local_distmat  # 融合局部距离

    # 3. 距离归一化（消除数值范围影响）
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))     # 按列归一化到[0,1]

    # 4. 初始化权重矩阵V和初始排名
    V = np.zeros_like(original_dist).astype(np.float16)  # 权重矩阵，存储样本间的相似度权重
    initial_rank = np.argsort(original_dist).astype(np.int32)  # 初始排名：每个样本的近邻排序（从小到大）

    # 5. 遍历每个样本，计算k-互近邻并构建权重矩阵V
    for i in range(all_num):
        # 步骤1：找样本i的前k1+1个近邻（forward k-neighbors）
        forward_k_neigh_index = initial_rank[i, :k1 + 1]

        # 步骤2：找这些近邻的前k1+1个近邻（backward k-neighbors）
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]

        # 步骤3：找“互近邻”——样本i是这些近邻的近邻（双向匹配）
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  # 样本i的k-互近邻

        # 步骤4：扩展互近邻（k-reciprocal expansion）
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            # 找候选样本的前k1/2个近邻（缩小范围，避免噪声）
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            # 找候选样本的互近邻
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            # 筛选：如果候选样本的互近邻和当前样本的互近邻重叠率>2/3，加入扩展列表
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        # 步骤5：去重，得到最终的扩展互近邻列表
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)

        # 步骤6：计算权重（距离越小，权重越大）
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])  # 指数衰减，距离小→权重大
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)  # 归一化权重
    # 6. 二次近邻平滑（k2>1时）
    original_dist = original_dist[:query_num, ]  # 截取query的初始距离
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            # 对每个样本i，取前k2个近邻的权重矩阵求平均，平滑噪声
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank  # 释放内存

    # 7. 构建逆索引：每个样本的非零权重样本列表（加速Jaccard计算）
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # 所有对样本i有非零权重的样本

    # 8. 计算Jaccard距离（衡量样本间的互近邻重叠度）
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]  # 样本i的非零权重样本
        indImages = [invIndex[ind] for ind in indNonZero]  # 这些样本的逆索引

        # 计算两两最小权重之和（Jaccard相似度的核心）
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        # Jaccard距离 = 1 - Jaccard相似度
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    # 9. 融合Jaccard距离和初始距离（lambda_value控制权重）
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value

    # 10. 释放内存 + 截取query-gallery的距离矩阵
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]  # 只保留query→gallery的距离
    return final_dist

