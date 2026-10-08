# -- coding:UTF-8
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import csv
import data_utils
import pandas as pd
from sklearn.manifold import TSNE
import argparse

BASE_FONT = 18

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = BASE_FONT
plt.rcParams['axes.titlesize'] = BASE_FONT + 4
plt.rcParams['axes.labelsize'] = BASE_FONT + 2
plt.rcParams['xtick.labelsize'] = BASE_FONT
plt.rcParams['ytick.labelsize'] = BASE_FONT
plt.rcParams['legend.fontsize'] = BASE_FONT - 1
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5

# Basic configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
herb_num = 563
gene_num = 2106
factor_num = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
herb_name_en = {
    33: 'Scutellaria barbata',
    11: 'Hedyotis diffusa',
    18: 'Paeonia lactiflora',
    182: 'Coptis chinensis',
    323: 'Artemisia annua',
    340: 'Panax notoginseng',
}

from data_utils import EnhancedHTI


# ==========================================
# Helper functions
# ==========================================
def load_unified_mapping(filepath):
    h_map, g_map = {}, {}
    if not os.path.exists(filepath):
        print(f"Warning: Mapping file {filepath} does not exist. IDs will be displayed")
        return h_map, g_map
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            name, idx, m_type = parts[0], int(parts[1]), parts[2]
            if m_type == 'herb':
                h_map[idx] = name
            else:
                g_map[idx] = name
    return h_map, g_map


# ==========================================
# General Model Loading
# ==========================================
def load_model(model_path):
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Weight file not found {model_path}")
        return None

    # Reconstruct data structures required by the model
    training_herb_set = np.load('./data/training_set.npy', allow_pickle=True)[0]
    training_gene_set = np.load('./data/training_set.npy', allow_pickle=True)[1]
    h_d = data_utils.readD(training_herb_set, herb_num)
    g_d = data_utils.readD(training_gene_set, gene_num)

    rel_hi = data_utils.readRelationalMatrices(training_herb_set, herb_num, gene_num, h_d, g_d)
    rel_ih = data_utils.readRelationalMatrices(training_gene_set, gene_num, herb_num, g_d, h_d)

    model = EnhancedHTI(herb_num, gene_num, factor_num, rel_hi, rel_ih, h_d, g_d, margin=0.5)
    model = model.to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded successfully")
    return model


# ==========================================
# Prediction Function
# ==========================================
def predict_all_herbs(model, top_k=10):
    """Predict target genes for all herbs"""
    print(f"Predict Top {herb_num} targets for all Top-{top_k} herbs...")
    predictions = {}
    with torch.no_grad():
        all_h, all_g = model(mode='predict')
        all_scores = torch.matmul(all_h, all_g.T)
        top_scores, top_indices = torch.topk(all_scores, top_k, dim=1)

        for h_id in range(herb_num):
            predictions[h_id] = {
                'target_indices': top_indices[h_id].cpu().numpy(),
                'scores': top_scores[h_id].cpu().numpy()
            }
    return predictions


def save_predictions_to_csv(predictions, herb_map, gene_map, filename='herb_target_predictions.csv'):
    """Save prediction results to CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Herb_ID', 'Herb_Name', 'Target_Rank', 'Target_Gene_ID', 'Target_Gene_Name', 'Score'])
        for h_id, pred in predictions.items():
            h_name = herb_map.get(h_id, f"Herb_{h_id}")
            for rank, (g_id, score) in enumerate(zip(pred['target_indices'], pred['scores']), 1):
                g_name = gene_map.get(g_id, f"Gene_{g_id}")
                writer.writerow([h_id, h_name, rank, g_id, g_name, f"{score:.6f}"])
    print(f"Full prediction results saved to: {filename}")


# ==========================================
# Chart Drawing Function
# ==========================================
def plot_target_network(predictions, gene_map, selected_herbs=[33, 11, 18], top_k=5, save_path='SCI_network_graph_optimized.png'):
    """Draw target prediction network graph"""
    print(f"Generating target interaction network graph (selected herbs: {selected_herbs})...")
    G = nx.Graph()

    for h_id in selected_herbs:
        h_name = herb_name_en.get(h_id, f"Herb_{h_id}")
        G.add_node(h_name, node_type='herb')
        top_genes = predictions[h_id]['target_indices'][:top_k]
        scores = predictions[h_id]['scores'][:top_k]

        for g_id, score in zip(top_genes, scores):
            g_name = gene_map.get(g_id, f"Gene_{g_id}")
            G.add_node(g_name, node_type='gene')
            G.add_edge(h_name, g_name, weight=score)

    plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    herb_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'herb']
    gene_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'gene']

    nx.draw_networkx_nodes(G, pos, nodelist=herb_nodes, node_color='#F08080',
                           node_size=3000, node_shape='s', edgecolors='white', linewidths=2.0)
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color='#98FB98',
                           node_size=3000, node_shape='o', edgecolors='white', linewidths=2.0)

    edges = G.edges()
    weights = [G[u][v]['weight'] * 0.4 for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    import matplotlib.patches as mpatches
    herb_patch = mpatches.Patch(color='#F08080', label='Herbs')
    target_patch = mpatches.Patch(color='#98FB98', label='Targets')
    plt.legend(handles=[herb_patch, target_patch], loc='upper right')

    plt.title('Herb-Target Interaction Network (Top Predictions)', fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"target network saved to {save_path}")


def plot_tsne_scatter(model, herb_limit=200, gene_limit=200, save_path='Fig5_tSNE_Visualization.png'):
    """Draw t-SNE scatter plot"""
    print("Generating t-SNE scatter plot...")
    try:
        model.eval()
        with torch.no_grad():
            all_h, all_g = model(mode='predict')
            herbs_emb = all_h[:herb_limit].cpu().numpy()
            genes_emb = all_g[:gene_limit].cpu().numpy()

        all_emb = np.vstack([herbs_emb, genes_emb])
        labels = ['Herb'] * len(herbs_emb) + ['Gene'] * len(genes_emb)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(all_emb)

        plt.figure(figsize=(10, 8), dpi=300)
        colors = {'Herb': '#2E86AB', 'Gene': '#C73E1D'}
        for label in ['Herb', 'Gene']:
            mask = np.array(labels) == label
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors[label], label=label, alpha=0.6, s=20)
        plt.title('t-SNE Visualization of Learned Embeddings',  fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"t-SNE scatter plot saved to {save_path}")
    except Exception as e:
        print(f"t-SNE plot generation failed: {e}")


def plot_violin(model, save_path='Fig_Score_Distribution_Violin.png'):
    """Draw violin plot of positive and negative sample score distribution"""
    print("Generating violin plot...")
    try:
        model.eval()
        with torch.no_grad():
            all_h, all_g = model(mode='predict')
            all_pre = torch.matmul(all_h, all_g.T).cpu().numpy()

        pos_scores = []
        neg_scores = []
        testing_herb_set = np.load('./data/testing_set.npy', allow_pickle=True)[0]
        all_rating = np.load('./data/set_all.npy', allow_pickle=True).item()
        set_all = set(range(gene_num))

        for u_i in testing_herb_set:
            gene_i_list = list(testing_herb_set[u_i])
            index_end_i = len(gene_i_list)
            gene_j_list = list(set_all - all_rating[u_i])

            gene_i_list.extend(gene_j_list)
            pre_one = all_pre[u_i][gene_i_list]

            pos_scores.extend(pre_one[:index_end_i])
            neg_candidates = pre_one[index_end_i:]
            sampled = np.random.choice(neg_candidates, size=min(50, len(neg_candidates)), replace=False)
            neg_scores.extend(sampled)

        df = pd.DataFrame({
            'Score': pos_scores + neg_scores,
            'Type': ['Positive'] * len(pos_scores) + ['Negative'] * len(neg_scores)
        })

        plt.figure(figsize=(8, 6), dpi=300)
        sns.violinplot(x='Type', y='Score', data=df, hue='Type', palette=['#2E86AB', '#C73E1D'], legend=False)
        plt.title('Score Distribution: Positive vs Negative Samples', fontweight='bold')
        plt.ylabel('Prediction Score')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Violin plot saved to {save_path}")
    except Exception as e:
        print(f"Failed to generate violin plot: {e}")


# ==========================================
# main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Light-HTINet-DNS predict and visualization')
    parser.add_argument('--seed', type=int, default=1, help='random seed (1, 42, 2024)')
    parser.add_argument('--epoch', type=int, default=399, help='Epoch checkpoint to load')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to model directory')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')

    args = parser.parse_args()

    # Determine Model Path
    if args.model_dir:
        model_path = os.path.join(args.model_dir, f'epoch{args.epoch}.pt')
    else:
        base_dir = f'./outputs/seed_{args.seed}/s_seed{args.seed}'
        model_path = os.path.join(base_dir, f'epoch{args.epoch}.pt')

    print(f"Configuration: seed={args.seed}, epoch={args.epoch}")
    print(f"Model Path: {model_path}")

    # Load Model
    model = load_model(model_path)
    if model is None:
        exit()

    # Load Mapping
    herb_map, gene_map = load_unified_mapping('./data/name2id.txt')

    # Prediction
    all_predictions = predict_all_herbs(model, top_k=10)

    # save CSV
    save_predictions_to_csv(all_predictions, herb_map, gene_map,
                            filename=os.path.join(args.output_dir, 'enhanced_herb_target_predictions.csv'))

    # Case Study: Multiple Herbs
    case_herbs = [33, 182]
    for target_herb_id in case_herbs:
        h_name = herb_map.get(target_herb_id, f"Herb_{target_herb_id}")
        print(f"\n[Case Study] herb: {h_name} (ID: {target_herb_id})")
        print("-" * 60)
        for rank, (g_id, score) in enumerate(zip(all_predictions[target_herb_id]['target_indices'],
                                                 all_predictions[target_herb_id]['scores']), 1):
            g_name = gene_map.get(g_id, f"Gene_{g_id}")
            print(f"  Rank {rank:2d}: {g_name:10s} (ID: {g_id:4d}) | Score: {score:.4f}")

    plot_tsne_scatter(model, save_path=os.path.join(args.output_dir, 'Fig5_tSNE_Visualization.png'))
    plot_violin(model, save_path=os.path.join(args.output_dir, 'Fig_Score_Distribution_Violin.png'))
    plot_target_network(all_predictions, gene_map, selected_herbs=[33, 182], top_k=5,
                        save_path=os.path.join(args.output_dir, 'SCI_network_graph_optimized.png'))
    print(f"Generated files are saved to: {args.output_dir}")