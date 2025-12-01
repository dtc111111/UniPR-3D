import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import vis_vars

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm

def do_opt(features, scores, dust_bin):
    p = get_matching_probs(scores, dust_bin, 3)
    p = torch.exp(p)
    # Normalize to maintain mass
    cluster_dim = features.shape[1] # features: B, cluster_dim, num_token
    num_clusters = scores.shape[1] # scores: B, num_clusters, num_token
    
    # dust_bin_score = p[:, -1, :] # B, num_token
    # vis_vars.vis_vars.append(dust_bin_score)

    p = p[:, :-1, :]
    p = p.unsqueeze(1).repeat(1, cluster_dim, 1, 1)
    f = features.unsqueeze(2).repeat(1, 1, num_clusters, 1)

    res = F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1) # B, cluster_dim * cluster_num
    return res


class TokenGeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    # x: (B, N, C)
    def forward(self, x):
        x = x.permute(0, 2, 1) # B, C, N
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1./self.p).squeeze(-1)
        # return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class TokenPooling(nn.Module):

    def __init__(self, 
                 dim_in: int = 2048,
                 dim_out: int = 256,
                 dim_pool: int = 512,
                 gem_p: float = 3.0,
                 dropout: float = 0.3,
                 eps: float = 1e-6):
        super().__init__()
        self.gem = TokenGeM(p=gem_p, eps=eps)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_pool = dim_pool
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()
        self.pre_proj = nn.Sequential(
            nn.Linear(self.dim_in, 1024),
            dropout,
            nn.ReLU(),
            nn.Linear(1024, self.dim_pool)
        )
        self.post_proj = nn.Sequential(
            nn.Linear(self.dim_pool, 512),
            dropout,
            nn.ReLU(),
            nn.Linear(512, self.dim_out)
        )
        

    def forward(self, x):
        # x: B, N, C_in)
        x = self.pre_proj(x) # B, N, C_pool
        x = F.normalize(x, p=2, dim=-1)
        x = self.gem(x) # B, C_pool
        x = self.post_proj(x) # B, C_out
        x = F.normalize(x, p=2, dim=-1)
        return x

class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x # Extract features and token

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]


        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)

        return F.normalize(f, p=2, dim=-1)


class SaladHead(nn.Module):

    def __init__(
        self,
        dim_in: int = 2048,
        dim_first: int = None,
        dim_register: int = None,
        dim_patch: int = None,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
        use_fisrt_token: bool = True,
        use_register_token: bool = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_first = dim_first if dim_first is not None else dim_in
        self.dim_register = dim_register if dim_register is not None else dim_in
        self.dim_patch = dim_patch if dim_patch is not None else dim_in
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.dropout = dropout
        self.use_fisrt_token = use_fisrt_token
        self.use_register_token = use_register_token

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        if self.use_fisrt_token:
            # MLP for global camera token
            self.camera_token_pooling = TokenPooling(dim_in=self.dim_first, dim_out=self.token_dim, dim_pool=512)

        if self.use_register_token:
            # MLP for global register token
            self.register_token_pooling = TokenPooling(dim_in=self.dim_register, dim_out=self.token_dim, dim_pool=512)

        # MLP for local features f_i
        self.patch_cluster_features = nn.Sequential(
            nn.Conv2d(self.dim_patch, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )

        # MLP for score matrix S
        self.patch_score = nn.Sequential(
            nn.Conv2d(self.dim_patch, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, aggregated_tokens_list: list, patch_start_idx: int):

        # Use Use tokens from the last block for pr task.
        tokens = aggregated_tokens_list[-1] # B, S, P, C
        
        if self.use_fisrt_token:
            camera_tokens = tokens[:, :, 0] # B, S, C
            camera_result = self.camera_token_pooling(camera_tokens)
        if self.use_register_token:
            register_tokens = tokens[:, :, 1:patch_start_idx]
            B, S, N, C = register_tokens.shape
            register_result = self.register_token_pooling(register_tokens.contiguous().view(B, S * N, C))

        patch_tokens = tokens[:, :, patch_start_idx:].permute(0, 3, 1, 2) # B, S, P, C -> B, C, S, P

        patch_features = self.patch_cluster_features(patch_tokens).flatten(2) # B, cluster_dim, S*P
        patch_scores = self.patch_score(patch_tokens).flatten(2) # B, cluster_num, S*P
        patch_result = do_opt(patch_features, patch_scores, self.dust_bin)

        result = [patch_result]

        if self.use_fisrt_token:
            result.append(camera_result)
        if self.use_register_token:
            result.append(register_result)
            
        return torch.cat(result, dim=-1)