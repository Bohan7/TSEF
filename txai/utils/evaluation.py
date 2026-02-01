import torch
from time import time as GETTIME
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from sklearn.metrics import explained_variance_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score
from scipy.stats import spearmanr
from tslearn.metrics import dtw_path


#### Metrics for attack
import torch
from time import time as GETTIME
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from sklearn.metrics import explained_variance_score, roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score
from scipy.stats import spearmanr
from tslearn.metrics import dtw_path
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import scipy
import scipy.stats as stats

def dtw_distance(x_adv, x):
    def l2(a, b):
        return np.linalg.norm(a - b)  # Euclidean between D-dim vectors

    T, B, d = x_adv.shape
    dtw_distances = []
    for i in range(B):
        xadv_norm = (x_adv[:, i, :] - x_adv[:, i, :].mean(axis=0, keepdims=True)) / (x_adv[:, i, :].std(axis=0, keepdims=True) + 1e-8)
        x_norm = (x[:, i, :] - x[:, i, :].mean(axis=0, keepdims=True)) / (x[:, i, :].std(axis=0, keepdims=True) + 1e-8)

        # result, path = fastdtw(x_adv[:, i, :], x[:, i, :], dist=l2, radius=2)
        result, path = fastdtw(xadv_norm, x_norm, dist=l2, radius=2)
        path_length = len(path)
        result = result  / path_length

        # Store the result
        dtw_distances.append(result)
    

    output_dict = {
        'dtw': dtw_distances,
    }

    return output_dict

def l2_distance(x_adv, x):
    T, B, d = x_adv.shape
    l2_distances = []
    for i in range(B):
        xadv_norm = (x_adv[:, i, :] - x_adv[:, i, :].mean(axis=0, keepdims=True)) / (x_adv[:, i, :].std(axis=0, keepdims=True) + 1e-8)
        x_norm = (x[:, i, :] - x[:, i, :].mean(axis=0, keepdims=True)) / (x[:, i, :].std(axis=0, keepdims=True) + 1e-8)


        # result = np.linalg.norm(x[:, i, :] - x_adv[:, i, :])
        result = np.linalg.norm(x_norm - xadv_norm)
        l2_distances.append(result)
    

    output_dict = {
        'l2_distance': l2_distances,
    }

    return output_dict

def mae_exp(generated_exps, gt_exps):
    T, B, d = generated_exps.shape
    l1_distances = []
    for i in range(B):
        result = np.linalg.norm(generated_exps[:, i, :].cpu().numpy() - gt_exps[:, i, :].cpu().numpy(), ord=1)
        l1_distances.append(result)
    

    output_dict = {
        'MAE_exp': l1_distances,
    }

    return output_dict

def rank_order_correlation(generated_exps, gt_exps, target_gt_exps):
    all_correlations = []
    target_all_correlations = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        correlation = scipy.stats.spearmanr(gte.cpu().numpy(), gene.cpu().numpy())[0]
        all_correlations.append(correlation)

        target_gte = target_gt_exps[:,i,:].flatten()
        target_correlation = scipy.stats.spearmanr(target_gte.cpu().numpy(), gene.cpu().numpy())[0]
        target_all_correlations.append(target_correlation)

    output_dict = {
        'rank_order_correlation': all_correlations,
        'target_rank_order_correlation': target_all_correlations,
    }

    return output_dict

def topk_intersection(generated_exps, gt_exps, target_gt_exps, k_top):
    all_topk_intersection = []
    target_all_topk_intersection = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        _, gte_indices = torch.topk(gte, k=k_top)
        _,  gene_indices = torch.topk(gene, k=k_top)
        intersection = float(len(np.intersect1d(gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        all_topk_intersection.append(intersection)

        target_gte = target_gt_exps[:,i,:].flatten()
        _, target_gte_indices = torch.topk(target_gte, k=k_top)
        target_intersection = float(len(np.intersect1d(target_gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        target_all_topk_intersection.append(target_intersection)

    output_dict = {
        'topk_intersection': all_topk_intersection,
        'target_topk_intersection': target_all_topk_intersection,
    }

    return output_dict

def topk_intersection_10(generated_exps, gt_exps, target_gt_exps, k_top):
    all_topk_intersection = []
    target_all_topk_intersection = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        _, gte_indices = torch.topk(gte, k=k_top)
        _,  gene_indices = torch.topk(gene, k=k_top)
        intersection = float(len(np.intersect1d(gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        all_topk_intersection.append(intersection)

        target_gte = target_gt_exps[:,i,:].flatten()
        _, target_gte_indices = torch.topk(target_gte, k=k_top)
        target_intersection = float(len(np.intersect1d(target_gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        target_all_topk_intersection.append(target_intersection)

    output_dict = {
        'topk_intersection_10': all_topk_intersection,
        'target_topk_intersection_10': target_all_topk_intersection,
    }

    return output_dict

def topk_intersection_30(generated_exps, gt_exps, target_gt_exps, k_top):
    all_topk_intersection = []
    target_all_topk_intersection = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        _, gte_indices = torch.topk(gte, k=k_top)
        _,  gene_indices = torch.topk(gene, k=k_top)
        intersection = float(len(np.intersect1d(gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        all_topk_intersection.append(intersection)

        target_gte = target_gt_exps[:,i,:].flatten()
        _, target_gte_indices = torch.topk(target_gte, k=k_top)
        target_intersection = float(len(np.intersect1d(target_gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        target_all_topk_intersection.append(target_intersection)

    output_dict = {
        'topk_intersection_30': all_topk_intersection,
        'target_topk_intersection_30': target_all_topk_intersection,
    }

    return output_dict

def topk_intersection_50(generated_exps, gt_exps, target_gt_exps, k_top):
    all_topk_intersection = []
    target_all_topk_intersection = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        _, gte_indices = torch.topk(gte, k=k_top)
        _,  gene_indices = torch.topk(gene, k=k_top)
        intersection = float(len(np.intersect1d(gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        all_topk_intersection.append(intersection)

        target_gte = target_gt_exps[:,i,:].flatten()
        _, target_gte_indices = torch.topk(target_gte, k=k_top)
        target_intersection = float(len(np.intersect1d(target_gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        target_all_topk_intersection.append(target_intersection)

    output_dict = {
        'topk_intersection_50': all_topk_intersection,
        'target_topk_intersection_50': target_all_topk_intersection,
    }

    return output_dict

def topk_intersection_5(generated_exps, gt_exps, k_top):
    all_topk_intersection = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        _, gte_indices = torch.topk(gte, k=k_top)
        _,  gene_indices = torch.topk(gene, k=k_top)
        intersection = float(len(np.intersect1d(gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        all_topk_intersection.append(intersection)

    output_dict = {
        'topk_intersection_5': all_topk_intersection,
    }

    return output_dict

def topk_intersection_threshold(generated_exps, gt_exps, threshold):
    all_topk_intersection = []

    for i in range(generated_exps.shape[1]):
        gte = gt_exps[:,i,:].flatten()
        gene = generated_exps[:,i,:].flatten()
        k_top = (gte > threshold).sum().item()

        if k_top == 0:
            print('ktop!!!!')
            continue
            # print(k_top / gte.size(0))

        _, gte_indices = torch.topk(gte, k=k_top)
        _,  gene_indices = torch.topk(gene, k=k_top)
        intersection = float(len(np.intersect1d(gte_indices.cpu().numpy(), gene_indices.cpu().numpy())))/k_top
        all_topk_intersection.append(intersection)

    output_dict = {
        'topk_intersection_threshold': all_topk_intersection,
    }

    return output_dict


def target_ground_truth_xai_eval(generated_exps, gt_exps, penalize_negatives = True, times = None):
    '''
    Compute auprc of generated explanation against ground-truth explanation
        - auprc is computed across one sample, averaged across all samples

    NOTE: Assumes all explanations have batch-first style, i.e. (B,T,d) - captum input/output

    Params:
        generated_exps: Explanations generated by method under evaluation
        gt_exps: Ground-truth explanations on which to evaluate
    '''

    # Normalize generated explanation:
    generated_exps = normalize_exp(generated_exps).detach().clone().cpu().numpy()
    gt_exps = gt_exps.detach().clone().cpu().numpy()
    gt_exps = gt_exps.astype(int)

    all_auprc, all_aup, all_aur = [], [], []

    for i in range(generated_exps.shape[1]):
        #auprc = roc_auc_score(gt_exps[i].flatten(), generated_exps[i].flatten())
        # print('gt exps', gt_exps[:,i,:].flatten().shape)
        # print('gen', generated_exps[:,i,:].flatten())
        if times is not None:
            gte = (gt_exps[:,i,:][times[:,i] > -1e5]).flatten()
            gene = normalize_one_exp(generated_exps[:,i,:][times[:,i] > -1e5]).flatten()
            auprc = average_precision_score(gte, gene)
            prec, rec, thres = precision_recall_curve(gte, gene)
        else:
            auprc = average_precision_score(gt_exps[:,i,:].flatten(), generated_exps[:,i,:].flatten())
            prec, rec, thres = precision_recall_curve(gt_exps[:,i,:].flatten(), generated_exps[:,i,:].flatten())

        # add for ig batch    
        if len(thres) >= 2:
            aur = auc(thres, rec[:-1]) # Last value in recall curve is always 0 (see sklearn documentation)
            aup = auc(thres, prec[:-1]) # Last value in precision curve is always 1 (see sklearn documentation)
        else:
            aur = 0.0
            aup = 0.0
        all_auprc.append(auprc)
        all_aup.append(aup)
        all_aur.append(aur)

    output_dict = {
        'target_auprc': all_auprc,
        'target_aup': all_aup,
        'target_aur': all_aur
    }

    return output_dict

def target_ground_truth_IoU(generated_exps, gt_exps, threshold = 0.9):
    '''
    Compute auprc of generated explanation against ground-truth explanation
        - auprc is computed across one sample, averaged across all samples

    NOTE: Assumes all explanations have batch-first style, i.e. (B,T,d) - captum input/output

    Params:
        generated_exps: Explanations generated by method under evaluation
        gt_exps: Ground-truth explanations on which to evaluate
    '''

    # Normalize generated explanation:
    generated_exps = normalize_exp(generated_exps).detach().clone().cpu()
    gt_exps = gt_exps.detach().clone().cpu().float()
    #gt_exps = gt_exps.astype(int)

    all_iou = []
    for i in range(generated_exps.shape[1]):
        #auprc = roc_auc_score(gt_exps[i].flatten(), generated_exps[i].flatten())
        # print('gt exps', gt_exps[:,i,:].flatten().shape)
        # print('gen', generated_exps[:,i,:].flatten())
        gen = generated_exps[:,i,:].flatten()
        thresh = torch.quantile(gen, threshold)
        generated_e = torch.where(gen >= thresh, 1.0, 0.0).float()
        iou = jaccard_similarity(generated_e, gt_exps[:,i,:].flatten())
        all_iou.append(iou)

    output_dict = {
        'target_iou': all_iou,
    }

    return output_dict




def faithfulness(model, X, time, y, pert_method, samples = 30):

    model.eval()
    reg_out = model(X, time).softmax()
    sum_faith = 0

    for s in samples:

        # Try to make perturbations until
        bad_pert = True
        while bad_pert:
            Xpert = pert_method(X)
            pert_out = model(Xpert, time).softmax()
            # check criteria
            criteria = (pert_out.argmax() == reg_out.argmax())
            bad_pert = not criteria # Bad perturbation if criteria does not hold
        
        sum_faith += F.kl_div(reg_out, pert_out).item()


    return (1 / samples) * sum_faith 

def similarity_faithfulness(model, test_X, test_time, explainer = None, explist = None, samples = 50):
    '''
    Measure faithfulness based on similarities of linear correlations
        between differences in 1) model prediction and 2) explanations

        - Use Dynamic Time Warping (DTW) for explanations
        - Use KL divergence for predictions

    If using explist feature, must provide testX, test_time that corresponds with indices in explist
    '''

    model.eval()

    # Sample pairs of samples
    if explainer is not None:
        inds = torch.arange(test_X.shape[1])
    else: # Use list
        inds = torch.arange(len(explist))

    combs = torch.combinations(inds)
    combs = combs[ torch.randperm(combs.shape[0])[:samples] ]

    exp_scores, pred_scores = [], []

    for i, j in tqdm(combs):

        predi = model(test_X[:,i,:], test_time[:,i].unsqueeze(dim=1))
        predj = model(test_X[:,j,:], test_time[:,j].unsqueeze(dim=1))

        # Get explanation for both:

        yi = predi.softmax(dim=1).argmax(dim=1) # Use computed y values (don't care about correctness)
        yj = predj.softmax(dim=1).argmax(dim=1)

        if explainer is not None:
            expi = explainer(model, test_X[:,i,:], test_time[:,i].unsqueeze(dim=1), y = yi).detach().clone().cpu().numpy()
            expj = explainer(model, test_X[:,j,:], test_time[:,j].unsqueeze(dim=1), y = yj).detach().clone().cpu().numpy()
        else: # Skip running the explainer, directly extract from tester
            expi = explist[int(i)]
            expj = explist[int(j)]

        cur = GETTIME()
        _, exp_sim = dtw_path(expi, expj)
        exp_scores.append(exp_sim)
        #exp_sim = torch.norm((expi - expj).flatten(), p = 2).item()

        predi, predj = F.log_softmax(predi, dim=1), F.log_softmax(predj, dim=1)

        pred_sim = F.kl_div(predi, predj, reduction = 'batchmean', log_target = True)
        pred_scores.append(pred_sim.item())

    print('Exp scores', exp_scores)
    print('Pred scores', pred_scores)
    cor, _ = spearmanr(exp_scores, pred_scores)

    return cor

def faithfulness_violation(model, X, times, mask_X, mask_times, y):
    '''
    Metric based on Faithfulness Violation presented in Liu et al., ICML 2022 
        (https://arxiv.org/pdf/2201.12114.pdf)
        - Don't use the violation part, only the change in confidence of label
    Params:
        model: Must only return prediction, no attention values, during forward call
        mask_X: 
        mask_times: 
    '''

    fullpred = model(X, times)
    maskpred = model(mask_X, mask_times)

    yind = y.int()
    # Must mask the 
    C = fullpred[0,yind] - maskpred[0,yind]

    return C

def normalize_exp(exps):
    norm_exps = torch.empty_like(exps)
    for i in range(exps.shape[1]):
        norm_exps[:,i,:] = (exps[:,i,:] - exps[:,i,:].min()) / (exps[:,i,:].max() - exps[:,i,:].min() + 1e-9)
    return norm_exps

def normalize_one_exp(exps):
    norm_exps = (exps - exps.min()) / (exps.max() - exps.min() + 1e-9)
    return norm_exps

def ground_truth_xai_eval(generated_exps, gt_exps, penalize_negatives = True, times = None):
    '''
    Compute auprc of generated explanation against ground-truth explanation
        - auprc is computed across one sample, averaged across all samples

    NOTE: Assumes all explanations have batch-first style, i.e. (B,T,d) - captum input/output

    Params:
        generated_exps: Explanations generated by method under evaluation
        gt_exps: Ground-truth explanations on which to evaluate
    '''

    # Normalize generated explanation:
    generated_exps = normalize_exp(generated_exps).detach().clone().cpu().numpy()
    gt_exps = gt_exps.detach().clone().cpu().numpy()
    gt_exps = gt_exps.astype(int)

    all_auprc, all_aup, all_aur = [], [], []

    for i in range(generated_exps.shape[1]):
        #auprc = roc_auc_score(gt_exps[i].flatten(), generated_exps[i].flatten())
        # print('gt exps', gt_exps[:,i,:].flatten().shape)
        # print('gen', generated_exps[:,i,:].flatten())
        if times is not None:
            gte = (gt_exps[:,i,:][times[:,i] > -1e5]).flatten()
            gene = normalize_one_exp(generated_exps[:,i,:][times[:,i] > -1e5]).flatten()
            auprc = average_precision_score(gte, gene)
            prec, rec, thres = precision_recall_curve(gte, gene)
        else:
            auprc = average_precision_score(gt_exps[:,i,:].flatten(), generated_exps[:,i,:].flatten())
            prec, rec, thres = precision_recall_curve(gt_exps[:,i,:].flatten(), generated_exps[:,i,:].flatten())
        if len(thres) >= 2:
            aur = auc(thres, rec[:-1]) # Last value in recall curve is always 0 (see sklearn documentation)
            aup = auc(thres, prec[:-1]) # Last value in precision curve is always 1 (see sklearn documentation)
        else:
            aur = 0.0
            aup = 0.0
        all_auprc.append(auprc)
        all_aup.append(aup)
        all_aur.append(aur)

    output_dict = {
        'auprc': all_auprc,
        'aup': all_aup,
        'aur': all_aur
    }

    return output_dict

def jaccard_similarity(tensor1, tensor2):
    # From ChatGPT
    intersection = torch.sum(tensor1 * tensor2)
    union = torch.sum(torch.max(tensor1, tensor2))  # Use max for element-wise OR
    
    jaccard = intersection / union
    
    return jaccard.item()

def ground_truth_IoU(generated_exps, gt_exps, threshold = 0.9):
    '''
    Compute auprc of generated explanation against ground-truth explanation
        - auprc is computed across one sample, averaged across all samples

    NOTE: Assumes all explanations have batch-first style, i.e. (B,T,d) - captum input/output

    Params:
        generated_exps: Explanations generated by method under evaluation
        gt_exps: Ground-truth explanations on which to evaluate
    '''

    # Normalize generated explanation:
    generated_exps = normalize_exp(generated_exps).detach().clone().cpu()
    gt_exps = gt_exps.detach().clone().cpu().float()
    #gt_exps = gt_exps.astype(int)

    all_iou = []
    for i in range(generated_exps.shape[1]):
        #auprc = roc_auc_score(gt_exps[i].flatten(), generated_exps[i].flatten())
        # print('gt exps', gt_exps[:,i,:].flatten().shape)
        # print('gen', generated_exps[:,i,:].flatten())
        gen = generated_exps[:,i,:].flatten()
        thresh = torch.quantile(gen, threshold)
        generated_e = torch.where(gen >= thresh, 1.0, 0.0).float()
        iou = jaccard_similarity(generated_e, gt_exps[:,i,:].flatten())
        all_iou.append(iou)

    output_dict = {
        'iou': all_iou,
    }

    return output_dict

def ground_truth_precision_recall(generated_exps, gt_exps, num_points = 50):
    '''
    Compute AUROC of generated explanation against ground-truth explanation
        - AUROC is computed across one sample, averaged across all samples

    NOTE: Assumes all explanations have batch-first style, i.e. (B,T,d) - captum input/output

    Params:
        generated_exps: Explanations generated by method under evaluation
        gt_exps: Ground-truth explanations on which to evaluate
    '''

    # Normalize generated explanation:
    generated_exps = normalize_exp(generated_exps).detach().clone().cpu().numpy()
    gt_exps = gt_exps.detach().clone().cpu().numpy().astype(int)

    thresholds = np.linspace(0,1,num_points)
    if num_points == 1:
        # Support for evaluating on masks
        thresholds = [0.5]

    total_prec, total_rec, masked_in = [], [], []

    for i in range(generated_exps.shape[0]):
         
        best_prec, best_rec, best_masked_in = -1, -1, -1

        # Try different thresholds for discretizing masks
        for t in thresholds:
            genexp = (generated_exps[i].flatten() > t).astype(int)
            prec = precision_score(gt_exps[i].flatten(), genexp)
            recall = recall_score(gt_exps[i].flatten(), genexp)

            # Determine if we have the best precision score:
            if best_prec < prec:
                best_prec = prec
                best_rec = recall
                best_masked_in = genexp.sum()

        total_prec.append(best_prec)
        total_rec.append(best_rec)
        masked_in.append(best_masked_in)

    return total_prec, total_rec, masked_in

def connected_component_count(exps):
    count = np.zeros(exps.shape[0])
    for i in range(exps.shape[0]): # Iterate over samples
        for j in range(exps.shape[2]): # Iterate over sensors
            on_bool = False
            for k in range(exps.shape[1]): # Iterate over sensors:
                if exps[i,k,j] == 0:
                    if on_bool: # If we hit a flip in the sensor reading
                        count[i] += 1
                        on_bool = False
                    # Nothing if we're already on False
                else:
                    on_bool = True # If we're now at 1, flip to True 

            else:
                if on_bool: # Upcount if we ended on on_bool
                    count[i] += 1

    return count # Should be length of samples

def count_time_sensor_sparsity(exps):
    '''
    Counts sparsity with respect to time points and sensors
    '''
    # Time first:
    timepts = exps.sum(dim=2)
    time_sparse = (timepts > 0).sum(dim=1) # Should be size of number of samples

    sensors = exps.sum(dim=1)
    sensor_sparse = (sensors > 0).sum(dim=1) # Should be size of number of samples

    return time_sparse.detach().clone().cpu().numpy(), sensor_sparse.detach().clone().cpu().numpy()

if __name__ == '__main__':
    # Test out connected components:
    a = torch.zeros(1, 50, 4)
    a[0,3:6,1] = 1
    a[0,4:9,2] = 1
    print(a)
    c = count_time_sensor_sparsity(a)
    print(c)