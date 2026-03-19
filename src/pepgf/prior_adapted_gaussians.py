import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    """
    Encodes EDN features into a style code for GEP conditioning.
    """
    def __init__(self, embed_dim=1024, style_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, style_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GlobalEntityPrior(nn.Module):
    """
    Frozen, class-specific priors for 3D Gaussian Splatting fields.
    Each class (building, road, etc.) has its own MLP outputting canonical Gaussians.
    """
    def __init__(self, num_classes=6, style_dim=64, num_points=1000):
        super().__init__()
        self.num_classes = num_classes
        self.style_dim = style_dim
        self.num_points = num_points
        
        # Each MLP outputs (mu(3), cov(6), op(1), sh(9)) for N points
        # For simplicity, we use one shared network with class embedding
        self.prior_network = nn.Sequential(
            nn.Linear(style_dim + 3 + num_classes, 256), # style + pos + class_onehot
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 + 6 + 1 + 9) # mu, cov_diag (or symm), op, sh
        )
        
        # Canonical points p_k in [-1, 1]^3
        self.register_buffer('canonical_points', torch.randn(num_points, 3))

    def forward(self, style_codes, class_labels):
        """
        Args:
            style_codes: (B, N_ent, style_dim)
            class_labels: (B, N_ent)
        Returns:
            mu, cov, op, sh: (B, N_ent, num_pts, ...)
        """
        B, N_ent, _ = style_codes.shape
        num_pts = self.num_points
        
        # One-hot classes
        classes_oh = F.one_hot(class_labels, num_classes=self.num_classes).float() # (B, N_ent, num_classes)
        
        # Prepare inputs for each entity and each point
        # style_codes: (B, N_ent, 1, style_dim)
        # points: (1, 1, num_pts, 3)
        # classes: (B, N_ent, 1, num_classes)
        s_rep = style_codes.unsqueeze(2).expand(-1, -1, num_pts, -1)
        p_rep = self.canonical_points.view(1, 1, num_pts, 3).expand(B, N_ent, -1, -1)
        c_rep = classes_oh.unsqueeze(2).expand(-1, -1, num_pts, -1)
        
        inputs = torch.cat([s_rep, p_rep, c_rep], dim=-1) # (B, N_ent, num_pts, dim)
        
        params = self.prior_network(inputs) # (B, N_ent, num_pts, 19)
        
        mu = params[..., 0:3]
        cov = params[..., 3:9]
        op = torch.sigmoid(params[..., 9:10])
        sh = params[..., 10:]
        
        return mu, cov, op, sh

class LocalResidualAdapter(nn.Module):
    """
    The learnable 'LoRA' part of the Gaussian field.
    Learns residuals ΔG relative to the frozen prior.
    """
    def __init__(self, num_points=1000):
        super().__init__()
        # Learnable deltas for each primitive in the entity
        # This is instantiated per entity in the scene
        self.mu_delta = nn.Parameter(torch.zeros(num_points, 3))
        self.cov_delta = nn.Parameter(torch.zeros(num_points, 6))
        self.op_delta = nn.Parameter(torch.zeros(num_points, 1))
        self.sh_delta = nn.Parameter(torch.zeros(num_points, 9))

    def forward(self):
        return self.mu_delta, self.cov_delta, self.op_delta, self.sh_delta

class PriorAdaptedGaussianFields(nn.Module):
    """
    The full PEPGF component.
    Orchestrates GEP (frozen) and LRA (trainable).
    """
    def __init__(self, num_classes=6, style_dim=64, num_points=1000):
        super().__init__()
        self.style_encoder = StyleEncoder(style_dim=style_dim)
        self.gep = GlobalEntityPrior(num_classes, style_dim, num_points)
        
        # Freeze GEP
        for param in self.gep.parameters():
            param.requires_grad = False
            
        # We store adapters in a ModuleDict mapped by entity_id
        # In a real training script, these are initialized per scene
        self.adapters = nn.ModuleDict() 

    def get_adapter(self, entity_id, num_points, device):
        if entity_id not in self.adapters:
            # Explicitly move new adapter to the same device as the model
            self.adapters[entity_id] = LocalResidualAdapter(num_points).to(device)
        return self.adapters[entity_id]

    def forward(self, entity_features, class_labels, entity_ids=None):
        """
        Args:
            entity_features: (B, N_ent, embed_dim)
            class_labels: (B, N_ent)
            entity_ids: list of strings or uniques per entity per batch
        """
        B, N_ent, _ = entity_features.shape
        
        # 1. Get style codes
        style_codes = self.style_encoder(entity_features)
        
        # 2. Get priors from GEP
        p_mu, p_cov, p_op, p_sh = self.gep(style_codes, class_labels)
        
        # 3. Add residuals from adapters
        # In batch mode, we assume we have access to local adapters
        # This might be tricky in pure batch, we use a placeholder additive logic
        if entity_ids is not None:
             res_mu, res_cov, res_op, res_sh = [], [], [], []
             for b in range(B):
                 b_mu, b_cov, b_op, b_sh = [], [], [], []
                 for i in range(N_ent):
                     # Unique ID for each entity in the world
                     eid = entity_ids[b][i]
                     adapter = self.get_adapter(eid, self.gep.num_points, entity_features.device)
                     rm, rc, ro, rs = adapter()
                     b_mu.append(rm)
                     b_cov.append(rc)
                     b_op.append(ro)
                     b_sh.append(rs)
                 res_mu.append(torch.stack(b_mu))
                 res_cov.append(torch.stack(b_cov))
                 res_op.append(torch.stack(b_op))
                 res_sh.append(torch.stack(b_sh))
             
             final_mu = p_mu + torch.stack(res_mu)
             final_cov = p_cov + torch.stack(res_cov)
             final_op = p_op + torch.stack(res_op)
             final_sh = p_sh + torch.stack(res_sh)
        else:
             final_mu, final_cov, final_op, final_sh = p_mu, p_cov, p_op, p_sh

        return final_mu, final_cov, final_op, final_sh

    def compute_prior_loss(self):
        """
        Regularization: ||ΔG||^2 to keep adapters close to prior.
        """
        loss = 0
        for adapter in self.adapters.values():
            loss += torch.norm(adapter.mu_delta)**2
            loss += torch.norm(adapter.cov_delta)**2
        return loss