# -- coding:UTF-8
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import time
import data_utils
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--output_dir', type=str, default='./model', help='Output directory for checkpoints')
parser.add_argument('--dns_k', type=int, default=12, help='DNS candidate pool size K')
parser.add_argument('--margin', type=float, default=0.5)
args = parser.parse_args()

SEED = args.seed
OUTPUT_DIR = args.output_dir

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)
print(f"Using random seed: {SEED}")
# ==========================================
# Environment setup
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================
# hyperparameter
# ==========================================
herb_num = 563
gene_num = 2106
factor_num = 64
batch_size = 2048 * 128

# run_id = f"exp2_seed{SEED}"

path_save_model_base = os.path.join(OUTPUT_DIR, f's_seed{SEED}')
if not os.path.exists(path_save_model_base):
    os.makedirs(path_save_model_base)

print("Loading data...")
training_herb_set, training_gene_set, training_set_count = np.load('./data/training_set.npy', allow_pickle=True)
# testing_herb_set, testing_gene_set, testing_set_count = np.load('./data/testing_set.npy', allow_pickle=True)
val_herb_set, val_gene_set, val_set_count = np.load('./data/val_set.npy', allow_pickle=True)
herb_rating_set_all = np.load('./data/set_all.npy', allow_pickle=True).item()

h_d = data_utils.readD(training_herb_set, herb_num)
g_d = data_utils.readD(training_gene_set, gene_num)

rel_hi = data_utils.readRelationalMatrices(training_herb_set, herb_num, gene_num, h_d, g_d)
rel_ih = data_utils.readRelationalMatrices(training_gene_set, gene_num, herb_num, g_d, h_d)

from data_utils import EnhancedHTI
model = EnhancedHTI(herb_num, gene_num, factor_num, rel_hi, rel_ih, h_d, g_d, margin=args.margin).to(device)

for r_id in rel_hi: rel_hi[r_id] = rel_hi[r_id].to(device)
for r_id in rel_ih: rel_ih[r_id] = rel_ih[r_id].to(device)

# ==========================================
# train_dataset = data_utils.BPRData(train_dict=training_herb_set, num_gene=gene_num, num_ng=5, is_training=True, data_set_count=training_set_count, all_rating=herb_rating_set_all, dns_k=12)
train_dataset = data_utils.BPRData(train_dict=training_herb_set, num_gene=gene_num, num_ng=5, is_training=True, data_set_count=training_set_count, all_rating=herb_rating_set_all, dns_k=args.dns_k)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset_loss = data_utils.BPRData(train_dict=val_herb_set, num_gene=gene_num, num_ng=5, is_training=True, data_set_count=val_set_count, all_rating=herb_rating_set_all, dns_k=1)
val_loader_loss = DataLoader(val_dataset_loss, batch_size=batch_size, shuffle=False, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print(f'-------Start clean training (Light-HTINet + DNS_K=12)-------')
# history_train_loss, history_val_loss, history_test_loss = [], [], []
history_train_loss, history_val_loss = [], []

for epoch in range(400):
    model.train()
    start_time = time.time()
    train_loader.dataset.ng_sample()
    train_loss_sum = []

    for herb, gene_i, gene_j in train_loader:
        herb, gene_i, gene_j = herb.to(device), gene_i.to(device), gene_j.to(device)
        optimizer.zero_grad()
        _, _, loss = model(herb, gene_i, gene_j, mode='train')
        loss.backward()
        optimizer.step()
        train_loss_sum.append(loss.item())

    elapsed_time = time.time() - start_time
    train_loss = round(np.mean(train_loss_sum), 4)

    if epoch % 10 == 0 or epoch == 399:
        torch.save(model.state_dict(), path_save_model_base + f'/epoch{epoch}.pt')

    model.eval()
    with torch.no_grad():
        val_loader_loss.dataset.ng_sample()
        val_loss = data_utils.metrics_loss(model, val_loader_loss, batch_size)
        # testing_loader_loss.dataset.ng_sample()
        # test_loss = data_utils.metrics_loss(model, testing_loader_loss, batch_size)

    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)
    print(f'Epoch: {epoch:3d} | Time: {round(elapsed_time, 1)}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# loss_filename = os.path.join(OUTPUT_DIR, f'real_loss_history_{run_id}.npz')
loss_filename = os.path.join(OUTPUT_DIR, 'loss_history.npz')

print(f'-------Training finished, saving loss data to{loss_filename}-------')
np.savez(loss_filename, train_loss=history_train_loss, val_loss=history_val_loss)
