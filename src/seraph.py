import torch
import torch.nn as nn
from src.edn import EntityDiscoveryNetwork
from src.hsg import HyperbolicGraph
from src.pepgf import PriorAdaptedGaussianFields
from src.gat import GlobalAssemblyTransformer
from src.renderer import DifferentiableRenderer

class SERAPH(nn.Module):
    """
    Semantic Entity Radiance with Adaptive Prior Hierarchy (SERAPH)
    
    The first Bayesian formulation of 3D reconstruction implemented end-to-end
    for locality-scale scenes.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else {}
        
        # Component 1: Entity Discovery Network (EDN)
        self.edn = EntityDiscoveryNetwork(
            num_entities=self.config.get('num_entities', 100),
            num_classes=self.config.get('num_classes', 6)
        )

        # Component 2: Hyperbolic Scene Graph (HSG) + HGNN
        self.hsg = HyperbolicGraph(
            embed_dim=1024, # from EDN backbone
            node_dim=self.config.get('hsg_node_dim', 128),
            num_layers=self.config.get('hsg_layers', 4)
        )

        # Component 3: Per-Entity Prior-Adapted Gaussian Fields (PEPGF)
        self.pepgf = PriorAdaptedGaussianFields(
            num_classes=self.config.get('num_classes', 6),
            style_dim=self.config.get('style_dim', 64),
            num_points=self.config.get('num_points', 1000)
        )

        # Component 4: Global Assembly Transformer (GAT)
        self.gat = GlobalAssemblyTransformer(
            embed_dim=1024,
            node_dim=self.config.get('hsg_node_dim', 128)
        )

        # Component 5: Differentiable Renderer (3DGS)
        self.renderer = DifferentiableRenderer(
            image_height=self.config.get('image_height', 1080),
            image_width=self.config.get('image_width', 1920)
        )

    def phase_1_organization(self, images):
        """
        Phase 1: Scene Organization (Per Locality)
        Discovers entities and optimizes hyperbolic layout/poses.
        """
        # 1. EDN: Entity Discovery
        # entity_features: (B, N, 1024), masks: (B, K, N, N_p), classes: (B, N, 6)
        entity_features, masks, classes = self.edn(images)
        
        # Image-level features for camera initialization (mean pool of patches)
        # Assuming masks is (B, K, N, N_p), we skip detailed masking for init
        camera_features = torch.mean(entity_features, dim=1, keepdim=True).repeat(1, images.size(1), 1)
        
        # 2. HSG: Layout and Poses
        # camera_poses: (B, K, 4, 4), entity_layout: (B, N, node_dim+1)
        camera_poses, entity_layout = self.hsg(entity_features, camera_features)
        
        # Layout Loss calculation (simplified)
        # visibility adjacency matrices could be derived from 'masks'
        adj_pos = torch.ones(images.size(0), images.size(1), entity_features.size(1)).to(images.device)
        adj_neg = torch.zeros_like(adj_pos)
        
        # camera_nodes are the nodes at index N: in h_joint
        # For layout loss, we need to extract them or pass them
        layout_loss = self.hsg.compute_layout_loss(
            entity_layout, 
            entity_layout, # placeholder for camera nodes
            adj_pos, 
            adj_neg
        )
        
        return {
            "entity_features": entity_features,
            "entity_layout": entity_layout,
            "camera_poses": camera_poses,
            "layout_loss": layout_loss,
            "entity_classes": classes,
            "entity_masks": masks
        }

    def phase_2_fine_tuning(self, images, p1_out):
        """
        Phase 2: Per-Entity Adapter Fine-tuning (Per Locality)
        Adapts local Gaussian fields to the specific scene residuals.
        """
        entity_features = p1_out["entity_features"]
        entity_layout = p1_out["entity_layout"]
        camera_poses = p1_out["camera_poses"]
        classes = torch.argmax(p1_out["entity_classes"], dim=-1) # (B, N)
        
        # Unique Entity IDs for this batch/locality
        # In a real scenario, these would come from the metadata
        B, N_ent, _ = entity_features.shape
        entity_ids = [[f"ent_{b}_{i}" for i in range(N_ent)] for b in range(B)]
        
        # 1. PEPGF: Adapt Gaussians
        # mu, cov, op, sh: (B, N_ent, num_pts, ...)
        entity_gaussians = self.pepgf(entity_features, classes, entity_ids=entity_ids)
        
        # 2. GAT: Hyperbolic Attention (Assembly)
        # Choosing first camera k=0 to render
        k = 0
        _, _, num_pts, _ = entity_gaussians[0].shape
        
        # Rendering queries (Simplified: image-grid ray query)
        # In actual gsplat/renderer, this is handled during rasterization.
        dummy_points = torch.randn(B, 1000, 3).to(images.device)
        dummy_dirs = torch.randn(B, 1000, 3).to(images.device)
        dummy_dist = torch.ones(B, 1000, 1).to(images.device)
        
        blending_weights = self.gat(entity_features, entity_layout, dummy_points, dummy_dirs, dummy_dist)
        
        # 3. Renderer: Novel View Synthesis
        rendered_image = self.renderer(entity_gaussians, blending_weights, camera_poses[:, k])
        
        return rendered_image

    def forward(self, images, mode="inference"):
        """
        Multimodal forward pass supporting three-phase training logic.
        """
        if mode == "train_organization":
            # Just Phase 1
            return self.phase_1_organization(images)
            
        elif mode == "train_fine_tuning":
            # Phase 1 setup + Phase 2 adaptation
            p1_out = self.phase_1_organization(images)
            rendered = self.phase_2_fine_tuning(images, p1_out)
            return rendered, p1_out
            
        else: # Inference
            with torch.no_grad():
                p1_out = self.phase_1_organization(images)
                rendered = self.phase_2_fine_tuning(images, p1_out)
                return rendered