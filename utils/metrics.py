import torch
import numpy as np
import os
from utils.reranking import re_ranking


# 欧氏距离  qf: query 特征 (num_q, dim)；  gf: gallery 特征 (num_g, dim)
# ∥q−g∥^2=∥q∥^2+∥g∥^2−2q⋅g
def euclidean_distance(qf, gf):
    # qf: query特征张量，形状[m, d] → m个query，每个特征d维
    # gf: gallery特征张量，形状[n, d] → n个gallery，每个特征d维
    # 返回：m×n的numpy距离矩阵（dist_mat[i,j] = 第i个query和第j个gallery的欧式距离平方）

    # 步骤1：获取样本数量
    m = qf.shape[0]  # query样本数m
    n = gf.shape[0]  # gallery样本数n

    # 步骤2：计算所有query的L2范数平方，并扩展为m×n矩阵
    # torch.pow(qf, 2)：对qf每个元素平方，形状[m, d]
    # .sum(dim=1, keepdim=True)：沿特征维度求和（计算L2范数平方），形状[m, 1]
    # .expand(m, n)：扩展为m×n矩阵（每行都是同一个query的范数平方）
    q_norm_sq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)

    # 步骤3：计算所有gallery的L2范数平方，扩展为m×n矩阵
    # torch.pow(gf, 2).sum(...)：形状[n, 1]
    # .expand(n, m)：扩展为n×m矩阵 → .t()：转置为m×n矩阵（每列都是同一个gallery的范数平方）
    g_norm_sq = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    # 步骤4：拼接前两项 → q_norm_sq + g_norm_sq（对应公式的||a||² + ||b||²）
    dist_mat = q_norm_sq + g_norm_sq

    # 步骤5：减去2×qf@gf.T（对应公式的-2a·b）
    # addmm_：原地执行 dist_mat = 1*dist_mat + (-2)*(qf @ gf.t())
    # qf@gf.t()：m×d 矩阵 × d×n 矩阵 = m×n矩阵（所有query和gallery的点积）
    dist_mat.addmm_(1, -2, qf, gf.t())

    # 步骤6：转到CPU，转为numpy（方便后续eval_func计算CMC/mAP）
    return dist_mat.cpu().numpy()

# 余弦距离
def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


# eval_func：真正计算 CMC 和 mAP 的地方
def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # 排序, 对每个 query，把 gallery 按距离从近到远排序。
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    # 正确匹配矩阵   得到一个 0/1 矩阵： 1：同一身份; 0：不同身份
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        # Market1501 规则:  同一个人 + 同一个摄像头 → 丢弃
        # ReID 的真实目标是跨摄像头找同一个人。不是：同一摄像头里找“同一张 / 连拍的图”,也不是“图像去重”
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query  # 查询集样本数量（如Market1501的num_query=3368）
        self.max_rank = max_rank  # CMC计算的最大排名（如50，即计算Rank-1到Rank-50）
        self.feat_norm = feat_norm  # 是否对特征归一化（ReID必须True，提升检索效果）
        self.reranking = reranking  # 是否使用重排序（提升mAP，计算成本更高）

    def reset(self):
        self.feats = []  # 存储所有样本的特征张量（list of tensor）
        self.pids = []  # 存储所有样本的行人ID（list of int）
        self.camids = []  # 存储所有样本的摄像头ID（list of int）

    def update(self, output):  # called once for each batch
        feat, pid, camid = output  # 接收单批次的特征、ID、摄像头ID
        self.feats.append(feat.cpu())  # 移到CPU，节省GPU显存
        self.pids.extend(np.asarray(pid))  # 转为numpy并追加到列表
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        # 步骤1：拼接所有批次的特征，得到完整特征矩阵
        feats = torch.cat(self.feats, dim=0)  # 形状[total_samples, D]

        # 步骤2：特征归一化（ReID必须！）
        if self.feat_norm:
            print("The test feature is normalized")
            # L2归一化：沿特征维度（dim=1），p=2表示L2范数
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # 步骤3：拆分query和gallery
        # query：前num_query个样本（待检索的查询集）
        qf = feats[:self.num_query]  # [num_query, D]
        # 用 np.asarray 转成 numpy 数组
        q_pids = np.asarray(self.pids[:self.num_query])  # [num_query]
        q_camids = np.asarray(self.camids[:self.num_query])  # [num_query]

        # gallery：剩余样本（候选图库集）
        gf = feats[self.num_query:]  # [num_gallery, D]
        g_pids = np.asarray(self.pids[self.num_query:])  # [num_gallery]
        g_camids = np.asarray(self.camids[self.num_query:])  # [num_gallery]

        # 步骤4：计算距离矩阵（query × gallery）
        if self.reranking:
            print('=> Enter reranking')
            # 重排序距离（更精准，速度慢）
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            # 欧式距离（速度快，基础版）
            distmat = euclidean_distance(qf, gf)  # 形状[num_query, num_gallery]

        # 步骤5：计算CMC和mAP
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # 返回结果（供日志打印）
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


