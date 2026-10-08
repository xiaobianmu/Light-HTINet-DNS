import torch
import os
import numpy as np
import time
import data_utils
from data_utils import EnhancedHTI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Path to model checkpoint directory')
args = parser.parse_args()

SEED = args.seed
path_save_model_base = args.model_dir

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

herb_num = 563
gene_num = 2106
factor_num = 64
top_ks = [1, 3, 5, 10]

print("Loading data...")
training_herb_set, training_gene_set, _ = np.load('./data/training_set.npy', allow_pickle=True)
testing_herb_set, testing_gene_set, _ = np.load('./data/testing_set.npy', allow_pickle=True)
val_herb_set, val_gene_set, _ = np.load('./data/val_set.npy', allow_pickle=True)
h_d = data_utils.readD(training_herb_set, herb_num)
g_d = data_utils.readD(training_gene_set, gene_num)

rel_hi = data_utils.readRelationalMatrices(training_herb_set, herb_num, gene_num, h_d, g_d)
rel_ih = data_utils.readRelationalMatrices(training_gene_set, gene_num, herb_num, g_d, h_d)

for r_id in rel_hi: rel_hi[r_id] = rel_hi[r_id].to(device)
for r_id in rel_ih: rel_ih[r_id] = rel_ih[r_id].to(device)

model = EnhancedHTI(herb_num, gene_num, factor_num, rel_hi, rel_ih, h_d, g_d, margin=0.5).to(device)
# ==========================================
def evaluate(model, eval_herb_set, training_herb_set, gene_num, top_ks):
    model.eval()
    with torch.no_grad():
        all_h, all_g = model(mode='predict')
        all_pre = torch.matmul(all_h, all_g.T).cpu().numpy()

    HR = {k: [] for k in top_ks}
    NDCG = {k: [] for k in top_ks}
    set_all = set(range(gene_num))

    for u_i in eval_herb_set:
        gene_i_list = list(eval_herb_set[u_i])
        index_end_i = len(gene_i_list)
        gene_j_list = list(set_all - training_herb_set[u_i] - eval_herb_set[u_i])
        gene_i_list.extend(gene_j_list)
        pre_one = all_pre[u_i][gene_i_list]

        max_k = max(top_ks)
        indices = list(data_utils.largest_indices(pre_one, max_k)[0])
        for k in top_ks:
            hr_t, ndcg_t = data_utils.hr_ndcg(indices[:k], index_end_i, k)
            HR[k].append(hr_t)
            NDCG[k].append(ndcg_t)

    result_HR = {k: round(np.mean(HR[k]), 4) for k in top_ks}
    result_NDCG = {k: round(np.mean(NDCG[k]), 4) for k in top_ks}
    return result_HR, result_NDCG

# ==========================================
print('\n-------- Step 1: Select the best on the validation set epoch --------')
epochs_to_test = list(range(0, 400, 10)) + [399]
best_val_hr10 = 0
best_epoch = None

for epoch in sorted(list(set(epochs_to_test))):
    PATH_model = path_save_model_base + f'/epoch{epoch}.pt'
    if not os.path.exists(PATH_model):
        continue
    try:
        model.load_state_dict(torch.load(PATH_model, map_location=device))
    except RuntimeError:
        continue

    val_HR, val_NDCG = evaluate(model, val_herb_set, training_herb_set, gene_num, top_ks)
    hr10 = val_HR[10]
    print(f'Epoch {epoch:3d} | Val HR@10: {hr10:.4f} | Val NDCG@10: {val_NDCG[10]:.4f}')

    if hr10 > best_val_hr10:
        best_val_hr10 = hr10
        best_epoch = epoch

print(f'\n validation best epoch = {best_epoch} (Val HR@10 = {best_val_hr10:.4f})')


# ==========================================
print('\n-------- Step 2: Evaluate the best epoch on the test set (only once). --------')
PATH_best = path_save_model_base + f'/epoch{best_epoch}.pt'
model.load_state_dict(torch.load(PATH_best, map_location=device))
test_HR, test_NDCG = evaluate(model, testing_herb_set, training_herb_set, gene_num, top_ks)

print("\n" + "=" * 50)
print(f" Best Epoch = {best_epoch}")
print("-" * 50)
for k in top_ks:
    print(f" HR@{k:<2}: {test_HR[k]:.4f}  |  NDCG@{k:<2}: {test_NDCG[k]:.4f}")
print("=" * 50)
