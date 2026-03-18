"""
Test script to verify SERAPH components work correctly
"""
import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_edn():
    """Test Entity Discovery Network"""
    print("Testing Entity Discovery Network...")
    from src.edn.model import EntityDiscoveryNetwork

    model = EntityDiscoveryNetwork(num_entities=20)
    dummy_input = torch.randn(1, 2, 3, 224, 224)  # batch=1, 2 images, 3 channels, 224x224

    try:
        features, masks, classes = model(dummy_input)
        print(f"✓ EDN works correctly")
        print(f"  Features shape: {features.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Classes shape: {classes.shape}")
        return True
    except Exception as e:
        print(f"✗ EDN test failed: {e}")
        return False

def test_hsg():
    """Test Hyperbolic Scene Graph"""
    print("Testing Hyperbolic Scene Graph...")
    from src.hsg.model import HyperbolicSceneGraph

    model = HyperbolicSceneGraph(node_dim=512, num_entities=20, num_cameras=3)
    B, N_entities, N_cameras = 1, 20, 3
    H, W = 224, 224

    entity_features = torch.randn(B, N_entities, 512)
    camera_features = torch.randn(B, N_cameras, 512)
    entity_masks = torch.rand(B, N_entities, H, W)

    try:
        camera_poses, entity_layout = model(entity_features, camera_features, entity_masks)
        print(f"✓ HSG works correctly")
        print(f"  Camera poses shape: {camera_poses.shape}")
        print(f"  Entity layout shape: {entity_layout.shape}")
        return True
    except Exception as e:
        print(f"✗ HSG test failed: {e}")
        return False

def test_pepgf():
    """Test Per-Entity Prior-Adapted Gaussian Fields"""
    print("Testing Per-Entity Prior-Adapted Gaussian Fields...")
    from src.pepgf.model import PerEntityPriorAdaptedGaussianFields

    model = PerEntityPriorAdaptedGaussianFields(num_classes=6, style_dim=32, num_gaussians=1000)
    B, N = 1, 20

    entity_features = torch.randn(B, N, 1024)
    class_labels = torch.randint(0, 6, (B, N))
    entity_poses = torch.randn(B, N, 6)

    try:
        gaussians, colors = model(entity_features, class_labels, entity_poses)
        print(f"✓ PEPGF works correctly")
        print(f"  Gaussians shape: {gaussians.shape}")
        print(f"  Colors shape: {colors.shape}")
        return True
    except Exception as e:
        print(f"✗ PEPGF test failed: {e}")
        return False

def test_gat():
    """Test Global Assembly Transformer"""
    print("Testing Global Assembly Transformer...")
    from src.gat.model import GlobalAssemblyTransformer

    model = GlobalAssemblyTransformer(num_entities=20, entity_dim=512)
    B, N = 1, 20
    H, W = 224, 224

    entity_features = torch.randn(B, N, 512)
    entity_masks = torch.rand(B, N, H, W)
    entity_poses = torch.randn(B, N, 6)

    try:
        scene_params, entity_params = model(entity_features, entity_masks, entity_poses)
        print(f"✓ GAT works correctly")
        print(f"  Scene params shape: {scene_params.shape}")
        print(f"  Entity params shape: {entity_params.shape}")
        return True
    except Exception as e:
        print(f"✗ GAT test failed: {e}")
        return False

def test_renderer():
    """Test Differentiable Renderer"""
    print("Testing Differentiable Renderer...")
    from src.renderer.model import DifferentiableRenderer

    model = DifferentiableRenderer(image_height=128, image_width=128, num_gaussians=500)
    B, N = 1, 500

    gaussians = torch.randn(B, N, 7)
    colors = torch.rand(B, N, 3)
    poses = torch.randn(B, 6)

    try:
        rendered_image, loss = model(gaussians, colors, poses)
        print(f"✓ Renderer works correctly")
        print(f"  Rendered image shape: {rendered_image.shape}")
        if loss is not None:
            print(f"  Loss computed: {loss.item()}")
        return True
    except Exception as e:
        print(f"✗ Renderer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running SERAPH component tests...\n")

    tests = [
        test_edn,
        test_hsg,
        test_pepgf,
        test_gat,
        test_renderer
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! SERAPH components are working correctly.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())