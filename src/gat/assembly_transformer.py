import torch
import torch.nn as nn
import torch.nn.functional as F
from src.hsg.hyperbolic_graph import LorentzManifold

class HyperbolicAttention(nn.Module):
    """
    Hyperbolic Attention for scale-adaptive LoD.
    Matches queries (rays/points) to entities in Lorentz space H^n.
    Distances from center (norm) determines the hierarchical depth.
    """
    def __init__(self, embed_dim, node_dim):
        super().__init__()
        self.manifold = LorentzManifold()
        # Project Euclidean queries to Lorentz manifold at the origin
        self.q_to_lorentz = nn.Linear(embed_dim, node_dim)

    def forward(self, query_tokens, entity_nodes):
        """
        Args:
            query_tokens: (B, N_q, embed_dim) - point-ray queries
            entity_nodes: (B, N_ent, node_dim+1) - hyperbolic layout
        Returns:
            attn_weights: (B, N_q, N_ent)
        """
        B, N_q, _ = query_tokens.shape
        
        # 1. Project queries to H^n
        v_q = self.q_to_lorentz(query_tokens)
        v_q = F.pad(v_q, (1, 0)) # v_0 = 0
        h_q = self.manifold.exp_map_origin(v_q)
        
        # 2. Compute hyperbolic distance d_L(h_q, h_ent)
        # distances between each query and each entity
        # h_q: (B, N_q, 1, dim+1)
        # h_ent: (B, 1, N_ent, dim+1)
        dists = self.manifold.dist(h_q.unsqueeze(2), entity_nodes.unsqueeze(1))
        
        # 3. Scale-adaptive attention
        # In hyperbolic space, dist depends on norm (depth).
        # Softmax over negative distances gives attention.
        attn_weights = F.softmax(-dists, dim=-1)
        
        return attn_weights

class GlobalAssemblyTransformer(nn.Module):
    """
    Assembles local entity-relative Gaussian fields into a globally consistent scene.
    Provides scale-adaptive rendering via hyperbolic attention.
    """
    def __init__(self, embed_dim=1024, node_dim=128):
        super().__init__()
        self.query_mlp = nn.Sequential(
            nn.Linear(3 + 3 + 1, 256), # [pos(3) || dir(3) || delta(1)]
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.hyp_attn = HyperbolicAttention(embed_dim, node_dim)

    def forward(self, entity_features, entity_layout, query_points, ray_dirs, camera_dist):
        """
        Args:
            entity_features: (B, N_ent, embed_dim)
            entity_layout: (B, N_ent, node_dim+1) in H^n
            query_points: (B, N_q, 3)
            ray_dirs: (B, N_q, 3)
            camera_dist: (B, N_q, 1) - delta from camera to point
        Returns:
            blending_weights: (B, N_q, N_ent)
        """
        # 1. Construct query tokens
        q_inputs = torch.cat([query_points, ray_dirs, camera_dist], dim=-1)
        q_tokens = self.query_mlp(q_inputs)
        
        # 2. Hyperbolic attention for LoD
        # Queries far from camera (large delta) -> small norm -> low-norm entities (roots)
        # Queries near camera (small delta) -> large norm -> high-norm entities (leaves)
        # This emerges from the learned query construction and attention.
        
        blending_weights = self.hyp_attn(q_tokens, entity_layout)
        
        return blending_weights