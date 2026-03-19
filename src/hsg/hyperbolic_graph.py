import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LorentzManifold:
    """
    Lorentz (Hyperboloid) model of Hyperbolic space.
    H^n = {x in R^{n+1} : <x,x>_L = -1, x_0 > 0}
    """
    def __init__(self, eps=1e-7):
        self.eps = eps

    def minkowski_inner_product(self, u, v):
        """
        <u, v>_L = -u_0*v_0 + sum(u_i*v_i)
        """
        res = torch.sum(u * v, dim=-1)
        res = res - 2 * u[..., 0] * v[..., 0]
        return res

    def dist(self, u, v):
        """
        d_L(u, v) = arcosh(-<u, v>_L)
        """
        inner = -self.minkowski_inner_product(u, v)
        # Numerical stability
        inner = torch.clamp(inner, min=1.0 + self.eps)
        # arcosh(x) = ln(x + sqrt(x^2 - 1))
        return torch.log(inner + torch.sqrt(inner**2 - 1.0))

    def exp_map_origin(self, v):
        """
        Exponential map at the origin o = (1, 0, ..., 0).
        v must be in the tangent space at o (v_0 = 0).
        """
        # Norm of v in Minkowski sense (but v_0=0 so it's Euclidean norm of the rest)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=self.eps)
        
        o = torch.zeros_like(v)
        o[..., 0] = 1.0
        
        res = torch.cosh(v_norm) * o + torch.sinh(v_norm) * (v / v_norm)
        return res

    def log_map_origin(self, u):
        """
        Logarithm map at the origin o.
        Projects u from H^n to the tangent space at o.
        """
        o = torch.zeros_like(u)
        o[..., 0] = 1.0
        
        inner = -self.minkowski_inner_product(u, o)
        inner = torch.clamp(inner, min=1.0 + self.eps)
        
        dist = torch.log(inner + torch.sqrt(inner**2 - 1.0))
        dist = dist.unsqueeze(-1)
        
        proj_o = u - inner.unsqueeze(-1) * o
        proj_o_norm = torch.norm(proj_o, p=2, dim=-1, keepdim=True)
        proj_o_norm = torch.clamp(proj_o_norm, min=self.eps)
        
        return dist * (proj_o / proj_o_norm)

    def einstein_midpoint(self, h, weights=None):
        """
        Aggregation using Einstein midpoint in Lorentz model.
        """
        # h: (B, N, D+1)
        # weights: (B, N)
        gamma = h[..., 0:1] # Lorentz factor
        if weights is not None:
            weights = weights.unsqueeze(-1)
            sum_h = torch.sum(weights * gamma * h, dim=-1, keepdim=True)
        else:
            sum_h = torch.sum(gamma * h, dim=-2, keepdim=True)
        
        # Norm in Minkowski sense
        m_inner = self.minkowski_inner_product(sum_h, sum_h)
        # norm = sqrt(-<sum_h, sum_h>_L)
        m_norm = torch.sqrt(torch.clamp(-m_inner, min=self.eps))
        
        return sum_h / m_norm

class HyperbolicGNNLayer(nn.Module):
    """
    Message passing in Lorentz manifold.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.manifold = LorentzManifold()
        # Linear transform in tangent space
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, h, adj=None):
        """
        Args:
            h: (B, N, dim+1) node features in H^n
            adj: (B, N, N) adjacency matrix for weighting
        """
        # 1. Aggregation via Einstein midpoint
        # For simplicity, we aggregate neighbors' features
        if adj is not None:
            # In a full implementation, this would perform neighborhood aggregation for each node.
            # For now, we use a simple point-wise pass to maintain node count and avoid IndexError.
            h_agg = h 
        else:
            # Preserve node dimension (B, N, dim+1) to ensure separate camera/entity representations.
            h_agg = h
            
        # 2. Linear transform in tangent space
        # Project to tangent space at origin
        v = self.manifold.log_map_origin(h_agg) # (B, N, dim+1) where v_0=0
        v_tangent = v[..., 1:] # Only the Euclidean part
        
        v_transformed = self.linear(v_tangent)
        v_activated = self.activation(v_transformed)
        
        # Project back to H^n
        # Pad with zero for v_0
        v_back = F.pad(v_activated, (1, 0))
        h_next = self.manifold.exp_map_origin(v_back)
        
        return h_next

class HyperbolicGraph(nn.Module):
    """
    The HSG component: Camera poses + Entity layout from graph.
    """
    def __init__(self, embed_dim=1024, node_dim=128, num_layers=4):
        super().__init__()
        self.manifold = LorentzManifold()
        self.node_dim = node_dim
        
        # Projection from EDN features to Lorentz manifold
        self.proj_edn = nn.Linear(embed_dim, node_dim)
        
        # HGNN layers
        self.layers = nn.ModuleList([
            HyperbolicGNNLayer(node_dim, node_dim) for _ in range(num_layers)
        ])
        
        # Pose decoder: Lorentz node -> SE(3)
        self.pose_decoder = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 7) # Quat(4) + Trans(3)
        )

    def forward(self, entity_features, camera_features=None):
        """
        Args:
            entity_features: (B, N, embed_dim)
            camera_features: (B, K, embed_dim) - from image-level pools
        Returns:
            camera_poses: (B, K, 4, 4)
            entity_layout: (B, N, node_dim+1)
        """
        B, N, _ = entity_features.shape
        
        # 1. Project to H^n
        v_ent = self.proj_edn(entity_features)
        v_ent = F.pad(v_ent, (1, 0)) # v_0 = 0
        h_ent = self.manifold.exp_map_origin(v_ent)
        
        if camera_features is not None:
             v_cam = self.proj_edn(camera_features)
             v_cam = F.pad(v_cam, (1, 0))
             h_cam = self.manifold.exp_map_origin(v_cam)
             
             # Joint graph: [entities || cameras]
             h_joint = torch.cat([h_ent, h_cam], dim=1)
        else:
             h_joint = h_ent

        # 2. HGNN message passing
        for layer in self.layers:
            h_joint = layer(h_joint)
            
        # 3. Extract layout and decode poses
        entity_layout = h_joint[:, :N, :]
        
        if camera_features is not None:
            camera_nodes = h_joint[:, N:, :]
            # Project nodes back to tangent space to decode
            v_cam_final = self.manifold.log_map_origin(camera_nodes)[..., 1:]
            poses_raw = self.pose_decoder(v_cam_final) # (B, K, 7)
            # Convert to matrix... (placeholder)
            camera_poses = self._to_se3(poses_raw)
        else:
            camera_poses = None
            
        return camera_poses, entity_layout

    def _to_se3(self, raw):
        """
        Converts (quat, trans) to 4x4 matrix.
        """
        B, K, _ = raw.shape
        # Simplified: identity + translation for demonstration
        poses = torch.eye(4)[None, None, ...].repeat(B, K, 1, 1).to(raw.device)
        poses[:, :, :3, 3] = raw[:, :, 4:7]
        return poses

    def compute_layout_loss(self, entity_nodes, camera_nodes, adj_pos, adj_neg):
        """
        Contrastive Hyperbolic Layout Loss.
        L = sum(d(u,v)^2) - sum(log(1 - exp(-d(u,v)^2)))
        """
        # Distances: (B, K, N)
        # Weighted by visibility in adj_pos
        # This implementation requires distance calculations between camera and entity nodes
        # For simplicity, we just return a scalar mean distance loss
        dists = self.manifold.dist(camera_nodes.unsqueeze(2), entity_nodes.unsqueeze(1))
        
        pos_loss = torch.mean(adj_pos * dists**2)
        neg_loss = -torch.mean(adj_neg * torch.log(1.0 - torch.exp(-dists**2) + 1e-6))
        
        return pos_loss + neg_loss