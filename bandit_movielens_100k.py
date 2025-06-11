#!/usr/bin/env python3
"""
bandit_movielens.py

Full pipeline:
 1) Load MovieLens data
 2) Per-user train/test split
 3) For each CF backbone (SVD / MultVAE / NCF):
    a) Compute item (V) and user (U) embeddings
    b) (Optional) L2-normalize embeddings
    c) Run Thompson Sampling bandit on (U,V)
 4) Plot average cumulative regret over time horizon for each backbone
"""

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- CF Backbones ----------------------------------------------------------
class MultVAE(nn.Module):
    def __init__(self, n_items, latent_dim=200):
        super().__init__()
        hidden_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mu, logvar


class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, hidden_layers=[64,32,16,8]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        layers = []
        dim = emb_dim * 2
        for h in hidden_layers:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, u, v):
        ue = self.user_emb(u)
        ve = self.item_emb(v)
        x = torch.cat([ue, ve], dim=1)
        return self.mlp(x).squeeze()


# -- CF Training / Embedding Extraction ------------------------------------
def svd_factorize(df, n_users, n_items, d):
    R = coo_matrix(
        (df.rating.values,
         (df.userId - 1, df.movieId - 1)),
        shape=(n_users, n_items)
    )
    svd = TruncatedSVD(n_components=d, random_state=0)
    U_s = svd.fit_transform(R)
    V_s = svd.components_.T
    S = np.diag(np.sqrt(svd.singular_values_))
    return U_s.dot(S), V_s.dot(S)


def train_multivae(df, n_users, n_items, latent_dim=200, epochs=20):
    print("Training MultVAE on device:", device)
    model = MultVAE(n_items, latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    user_items = df.groupby('userId')['movieId'].apply(lambda x: x.values - 1)

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        model.train()
        for items in tqdm(user_items, desc="MultVAE users", unit="user"):
            x = torch.zeros(1, n_items, device=device)
            x[0, items] = 1.0
            recon, mu, logvar = model(x)
            recon_loss = - (x * torch.log(recon + 1e-10)).sum()
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld
            opt.zero_grad()
            loss.backward()
            opt.step()

    print("Extracting embeddings...")
    # decoder[2] is the final linear layer; weight shape = (n_items, latent_dim)
    V = model.decoder[2].weight.data.cpu().numpy()

    U = []
    model.eval()
    with torch.no_grad():
        for u in tqdm(range(1, n_users+1), desc="MultVAE embed", unit="user"):
            idxs = df[df.userId == u].movieId.values - 1
            x = torch.zeros(1, n_items, device=device)
            x[0, idxs] = 1.0
            stats = model.encoder(x)
            mu, _ = stats.chunk(2, dim=1)
            U.append(mu.squeeze().cpu().numpy())

    return np.vstack(U), V


def train_ncf(df, n_users, n_items, emb_dim=32, epochs=5):
    print("Training NCF on device:", device)
    model = NCF(n_users, n_items, emb_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    users = torch.tensor(df.userId.values - 1, dtype=torch.long, device=device)
    items = torch.tensor(df.movieId.values - 1, dtype=torch.long, device=device)
    ratings = torch.tensor(df.rating.values, dtype=torch.float, device=device)

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        model.train()
        opt.zero_grad()
        preds = model(users, items)
        loss = loss_fn(preds, ratings)
        loss.backward()
        opt.step()

    print("Extracting embeddings...")
    U = model.user_emb.weight.data.cpu().numpy()
    V = model.item_emb.weight.data.cpu().numpy()
    return U, V


# -- Data splitting ---------------------------------------------------------
def load_and_split(path, hold=0.2):
    df = pd.read_csv(path, sep="\t", names=["userId", "movieId", "rating", "ts"])
    mask = np.ones(len(df), dtype=bool)
    for u, idxs in df.groupby('userId').groups.items():
        k = int(len(idxs) * hold)
        rem = np.random.choice(idxs, k, replace=False)
        mask[rem] = False
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


# -- Thompson Sampling Agent -----------------------------------------------
class LinTS:
    def __init__(self, d, nu=1.0, init_jitter=1e-6, jitter_factor=10.0):
        self.nu           = nu
        self.init_jitter  = init_jitter
        self.jitter_factor= jitter_factor
        self.A_inv        = np.eye(d)       # posterior covariance
        self.b            = np.zeros(d)     # posterior mean numerator

    def select(self, V):
        mu  = self.A_inv.dot(self.b)
        cov = self.nu**2 * self.A_inv

        # 1) symmetrize
        cov = 0.5 * (cov + cov.T)
        # 2) add initial jitter
        jitter = self.init_jitter
        cov += jitter * np.eye(cov.shape[0])

        # 3) ensure SPD by eigen-clipping
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            # fallback: add more jitter and retry eigh
            cov += jitter * self.jitter_factor * np.eye(cov.shape[0])
            eigvals, eigvecs = np.linalg.eigh(cov)

        # clip any tiny negative eigenvalues to zero
        eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
        # rebuild a valid SPD covariance
        cov_spd = (eigvecs * eigvals_clipped) @ eigvecs.T

        # 4) sample θ ∼ N(mu, cov_spd)
        try:
            theta = np.random.multivariate_normal(mu, cov_spd)
        except np.linalg.LinAlgError:
            # final fallback to diagonal-only
            std_diag = np.sqrt(eigvals_clipped)
            theta = mu + eigvecs @ (std_diag * np.random.randn(len(std_diag)))

        # pick best arm
        return np.argmax(V.dot(theta))

    def update(self, x, r):
        # Sherman–Morrison rank-1 update of A_inv
        Axi   = self.A_inv.dot(x)
        denom = 1.0 + x.dot(Axi)
        self.A_inv -= np.outer(Axi, Axi) / denom
        self.b     += r * x
        

# -- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',      required=True, help='Path to u.data')
    parser.add_argument('--hold',      type=float, default=0.2)
    parser.add_argument('--T',         type=int,   default=500)
    parser.add_argument('--noise',     type=float, default=0.1)
    parser.add_argument('--nu',        type=float, default=1.0,
                        help='Thompson sampling scale')
    parser.add_argument('--normalize', action='store_true',
                        help='L2-normalize U and V')
    args = parser.parse_args()

    # 1) Load & split once
    train_df, _ = load_and_split(args.path, args.hold)
    df_full     = pd.read_csv(args.path, sep="\t",
                              names=["userId","movieId","rating","ts"])
    n_u, n_i    = df_full.userId.max(), df_full.movieId.max()

    # embedding dimensions to sweep over
    dims = [20, 50, 100]

    # iterate over dims
    for d in dims:
        print(f"\n=== Embedding dimension: {d} ===")
        backbones = ['svd', 'multivae', 'ncf']
        regret_curves = {m: np.zeros(args.T) for m in backbones}

        # helper to get embeddings at this dimension
        def get_embeddings(model_name):
            if model_name == 'svd':
                return svd_factorize(train_df, n_u, n_i, d)
            elif model_name == 'multivae':
                return train_multivae(train_df, n_u, n_i, latent_dim=d)
            else:
                return train_ncf(train_df, n_u, n_i, emb_dim=d)
        
        # optional L2-normalization
        def normalize_feats(X, eps=1e-8):
            # compute L2 norms
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # avoid zeros: any zero-norm becomes 1.0 (so row stays zero), or add eps
            norms = np.where(norms < eps, 1.0, norms)
            return X / norms

        # run Thompson sampling for each backbone
        for model_name in backbones:
            print(f" → Backbone: {model_name}")
            U, V = get_embeddings(model_name)

            if args.normalize:
                U = normalize_feats(U)
                V = normalize_feats(V)

            for u in tqdm(range(n_u),
                          desc=f"Users ({model_name}, d={d})",
                          unit="user"):
                agent   = LinTS(d=V.shape[1], nu=args.nu)
                best    = U[u].dot(V.T).max()
                cum_reg = 0.0

                for t in range(args.T):
                    a = agent.select(V)
                    r = U[u].dot(V[a]) + np.random.randn()*args.noise
                    agent.update(V[a], r)
                    cum_reg += (best - r)
                    regret_curves[model_name][t] += cum_reg

            # average over users
            regret_curves[model_name] /= n_u

        # plot
        plt.figure()
        for m in backbones:
            plt.plot(np.arange(1, args.T+1),
                     regret_curves[m],
                     label=m)
        plt.xlabel("Time step")
        plt.ylabel("Avg cumulative regret")
        plt.title(f"Thompson Sampling on CF Backbones (d={d})")
        plt.legend()
        plt.tight_layout()
        outname = f"ps_compare_backbones_dim{d}.png"
        plt.savefig(outname)
        print(f" → Saved plot: {outname}")
        plt.close()

    print("\nAll done.")

if __name__ == '__main__':
    main()