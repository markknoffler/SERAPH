import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange, repeat

class DINOv2Backbone(nn.Module):
    """
    Frozen DINOv2 backbone for semantically coherent patch features.
    """
    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m', out_dim=1024):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_embed.patch_size[0]

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            patch_tokens: (B, N_p, embed_dim)
        """
        # x is assumed to be normalized
        tokens = self.model.forward_features(x)
        # tokens usually (B, 1 + N_p, embed_dim) including CLS
        patch_tokens = tokens[:, 1:, :] 
        return patch_tokens

class CrossImageAttentionLayer(nn.Module):
    """
    Multi-head attention across all input images.
    """
    def __init__(self, embed_dim, num_heads=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B*K, N_p, embed_dim) or flattened across K
        """
        # Here we assume x is (B, K * N_p, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class EntityClusteringHead(nn.Module):
    """
    Differentiable k-means for soft entity assignment.
    """
    def __init__(self, embed_dim, num_entities=100):
        super().__init__()
        self.num_entities = num_entities
        self.cluster_centroids = nn.Parameter(torch.randn(num_entities, embed_dim))
        self.temperature = 0.1

    def forward(self, tokens):
        """
        Args:
            tokens: (B, Total_Tokens, embed_dim)
        Returns:
            assignment: (B, Total_Tokens, num_entities) - Soft assignment matrix
            centroids: (B, num_entities, embed_dim) - Updated centroids
        """
        # Soft assignment using cosine similarity
        # Similarity: (B, Total_Tokens, num_entities)
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        centroids_norm = F.normalize(self.cluster_centroids, p=2, dim=-1)
        
        sim = torch.matmul(tokens_norm, centroids_norm.transpose(0, 1)) / self.temperature
        assignment = F.softmax(sim, dim=-1) # (B, Total_Tokens, N)
        
        # Weighted mean to get entity features
        # (B, N, Total_Tokens) @ (B, Total_Tokens, embed_dim) -> (B, N, embed_dim)
        entity_features = torch.matmul(assignment.transpose(1, 2), tokens)
        
        # Normalize by total weight
        weights = assignment.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6
        entity_features = entity_features / weights
        
        return assignment, entity_features

class SemanticClassificationHead(nn.Module):
    """
    Predicts semantic class for each entity.
    """
    def __init__(self, embed_dim, num_classes=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, entity_features):
        return self.mlp(entity_features)

class EntityDiscoveryNetwork(nn.Module):
    """
    The full EDN component.
    """
    def __init__(self, num_entities=100, num_classes=6):
        super().__init__()
        self.backbone = DINOv2Backbone()
        self.cross_attn = CrossImageAttentionLayer(self.backbone.embed_dim)
        self.clustering = EntityClusteringHead(self.backbone.embed_dim, num_entities)
        self.classifier = SemanticClassificationHead(self.backbone.embed_dim, num_classes)

    def forward(self, images):
        """
        Args:
            images: (B, K, 3, H, W) where K is number of views
        Returns:
            entity_features: (B, N, embed_dim)
            masks: (B, K, N, H, W) - soft visibility masks
            classes: (B, N, num_classes) - semantic logits
        """
        B, K, C, H, W = images.shape
        # Flatten B, K for backbone
        x = rearrange(images, 'b k c h w -> (b k) c h w')
        patch_tokens = self.backbone(x) # (B*K, N_p, embed_dim)
        
        # Cross-image attention
        # Flatten all patches across views for a single batch
        # (B, K * N_p, embed_dim)
        tokens_all = rearrange(patch_tokens, '(b k) n d -> b (k n) d', b=B, k=K)
        tokens_refined = self.cross_attn(tokens_all)
        
        # Clustering
        assignment, entity_features = self.clustering(tokens_refined) # assignments: (B, K*N_p, N)
        
        # Classification
        classes = self.classifier(entity_features)
        
        # Reshape assignments back to masks
        # Patch-level assignments: (B, K, N_p, N)
        patch_masks = rearrange(assignment, 'b (k n) e -> b k n e', k=K)
        # We can upscale or just leave it as patch-level masks for now
        # For simplicity, returning (B, K, N, N_p)
        masks = patch_masks.transpose(2, 3) 
        
        return entity_features, masks, classes