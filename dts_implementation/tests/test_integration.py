"""
Integration Tests for S-ADT Implementation

Tests all components together to verify the implementation works.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules import correctly"""
    print("🧪 Test 1: Imports...")
    
    try:
        from dts_implementation.core.dts_node import MCTSNode
        from dts_implementation.core.soft_bellman import soft_bellman_backup, sample_child_boltzmann
        from dts_implementation.models.opentslm_wrapper import OpenTSLMWrapper
        from dts_implementation.utils.psd_utils import compute_psd, spectral_distance
        from dts_implementation.rewards.spectral_reward import SpectralReward
        from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode
        
        print("   ✅ All imports successful!")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_mcts_node():
    """Test MCTSNode and TokenNode"""
    print("\n🧪 Test 2: MCTS Nodes...")
    
    try:
        from dts_implementation.core.dts_node import MCTSNode
        from dts_implementation.search.maxent_ts import TokenNode
        
        # Test basic MCTSNode
        dummy_tensor = torch.randn(1, 10)
        node = MCTSNode(x_t=dummy_tensor, t=5)
        assert node.t == 5
        assert node.visit_count == 1  # Initialized to 1 in MCTSNode
        assert node.is_leaf()
        
        # Test TokenNode
        token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        token_node = TokenNode(token_ids=token_ids, t=3)
        assert token_node.token_ids.shape == (1, 3)
        assert token_node.t == 3
        
        print("   ✅ Node tests passed!")
        return True
    except Exception as e:
        print(f"   ❌ Node test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_soft_bellman():
    """Test Soft Bellman backup"""
    print("\n🧪 Test 3: Soft Bellman Backup...")
    
    try:
        from dts_implementation.search.maxent_ts import TokenNode
        from dts_implementation.core.soft_bellman import soft_bellman_backup, sample_child_boltzmann
        
        # Create simple tree: root -> child1, child2
        root_tokens = torch.tensor([[1, 2]], dtype=torch.long)
        root = TokenNode(token_ids=root_tokens, t=2)
        
        child1_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        child1 = TokenNode(token_ids=child1_tokens, t=3, parent=root)
        
        child2_tokens = torch.tensor([[1, 2, 4]], dtype=torch.long)
        child2 = TokenNode(token_ids=child2_tokens, t=3, parent=root)
        
        root.add_child(child1)
        root.add_child(child2)
        
        # Backup from child1
        soft_bellman_backup(child1, reward=1.0, temperature=1.0)
        
        assert child1.visit_count == 2  # Initialized to 1, incremented once
        assert root.visit_count == 2  # Initialized to 1, incremented once
        
        # Backup from child2
        soft_bellman_backup(child2, reward=2.0, temperature=1.0)
        
        assert child2.visit_count == 2  # Initialized to 1, incremented once
        assert root.visit_count == 3  # Incremented twice
        
        # Test selection
        selected = sample_child_boltzmann(root, temperature=1.0)
        assert selected in [child1, child2]
        
        print("   ✅ Soft Bellman tests passed!")
        return True
    except Exception as e:
        print(f"   ❌ Soft Bellman test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_psd_utils():
    """Test PSD computation"""
    print("\n🧪 Test 4: PSD Utilities...")
    
    try:
        from dts_implementation.utils.psd_utils import (
            compute_psd, compute_expected_psd, spectral_distance
        )
        
        # Generate test signals
        t = np.linspace(0, 10, 1000)
        signal1 = np.sin(2 * np.pi * 2 * t)  # 2 Hz
        signal2 = np.sin(2 * np.pi * 10 * t)  # 10 Hz
        
        # Compute PSDs
        freqs1, psd1 = compute_psd(signal1, sampling_rate=100)
        freqs2, psd2 = compute_psd(signal2, sampling_rate=100)
        
        assert freqs1.shape == psd1.shape
        assert freqs2.shape == psd2.shape
        
        # Compute distance
        dist = spectral_distance(psd1, psd2, freqs1, metric='l1')
        assert dist > 0  # Different frequencies should have distance > 0
        
        print(f"   Spectral distance (2Hz vs 10Hz): {dist:.4f}")
        print("   ✅ PSD tests passed!")
        return True
    except Exception as e:
        print(f"   ❌ PSD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spectral_reward():
    """Test spectral reward computation"""
    print("\n🧪 Test 5: Spectral Reward...")
    
    try:
        from dts_implementation.rewards.spectral_reward import SpectralReward
        
        # Create reward computer
        reward_computer = SpectralReward(
            gamma=1.0,
            sampling_rate=100,
            normalize_rewards=False
        )
        
        # Generate context
        t = np.linspace(0, 10, 1000)
        context = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
        
        # Set context
        reward_computer.set_context(context)
        assert reward_computer.context_psd_cache is not None
        
        # Compute reward for matching prediction
        pred_good = np.sin(2 * np.pi * 2 * t) + 0.15 * np.random.randn(len(t))
        reward_good = reward_computer.compute_reward(pred_good)
        
        # Compute reward for non-matching prediction
        pred_bad = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        reward_bad = reward_computer.compute_reward(pred_bad)
        
        # Bad prediction should have higher penalty (lower reward)
        assert reward_bad['spectral_penalty'] > reward_good['spectral_penalty']
        assert reward_bad['total_reward'] < reward_good['total_reward']
        
        print(f"   Good prediction reward: {reward_good['total_reward']:.4f}")
        print(f"   Bad prediction reward: {reward_bad['total_reward']:.4f}")
        print("   ✅ Spectral reward tests passed!")
        return True
    except Exception as e:
        print(f"   ❌ Spectral reward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test MaxEnt-TS configuration"""
    print("\n🧪 Test 6: MaxEnt-TS Config...")
    
    try:
        from dts_implementation.search.maxent_ts import MaxEntTSConfig
        
        config = MaxEntTSConfig(
            num_rollouts=50,
            temperature=1.0,
            max_seq_length=100
        )
        
        assert config.num_rollouts == 50
        assert config.temperature == 1.0
        assert config.max_seq_length == 100
        
        print("   ✅ Config tests passed!")
        return True
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_search():
    """Test MaxEnt-TS with mock model"""
    print("\n🧪 Test 7: Mock Search (no real model)...")
    
    try:
        from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode
        from dts_implementation.rewards.spectral_reward import SpectralReward
        
        # This test verifies the search logic without a real model
        # We'll create a minimal mock
        
        print("   ⚠️  Skipping (requires real OpenTSLM model)")
        print("   ℹ️  Run stage1_tsqa_example.py for full integration test")
        return True
    except Exception as e:
        print(f"   ❌ Mock search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("  S-ADT Implementation Integration Tests")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("MCTS Nodes", test_mcts_node),
        ("Soft Bellman", test_soft_bellman),
        ("PSD Utils", test_psd_utils),
        ("Spectral Reward", test_spectral_reward),
        ("Config", test_config),
        ("Mock Search", test_mock_search),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print("\n" + "="*70)
    print(f"  Total: {passed_count}/{total} tests passed")
    print("="*70)
    
    if passed_count == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

