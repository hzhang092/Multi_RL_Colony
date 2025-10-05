#!/usr/bin/env python3
"""
Simple Colony Visualization Script

A simple wrapper script to quickly visualize a trained agent without command-line arguments.
This script automatically finds the latest checkpoint and runs the visualization.

Usage:
    python visualize_agent.py

Features:
- Automatically finds and loads the latest trained model
- Real-time colony growth visualization
- No command-line arguments needed
- Uses deterministic actions for consistent behavior

"""

import sys
from pathlib import Path

# Add the parent directory to path to access other modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluations.ppo_eval import run_evaluation, find_latest_checkpoint


def main():
    """
    Simple main function that runs the visualization with default settings.
    """
    print("🔬 Colony Growth Visualization")
    print("=" * 50)
    
    try:
        # Find the latest checkpoint
        checkpoint_path = find_latest_checkpoint()
        print(f"📁 Using checkpoint: {Path(checkpoint_path).name}")
        
        # Run evaluation with visualization
        print("🚀 Starting visualization...")
        results = run_evaluation(
            checkpoint_path=checkpoint_path,
            render=True,           # Enable visualization
            max_steps=100,         # Run for up to 100 steps
            deterministic=True,    # Use deterministic actions
            save_frames=False,     # Don't save frames by default
            frame_interval=3,      # Update every 3 steps
            env_seed=686          # Fixed seed for reproducibility
        )
        
        print("\n🎉 Visualization completed!")
        print(f"   Final colony size: {results['final_num_cells']} cells")
        print(f"   Total reward: {results['total_reward']:.3f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have trained a model first by running:")
        print("   python experiments/ppo_train.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Visualization interrupted by user")
        
    except Exception as e:
        print(f"❌ Rendering error encountered: {e}")
        print("💡 Trying evaluation without visualization...")
        
        try:
            # Retry without rendering
            checkpoint_path = find_latest_checkpoint()
            results = run_evaluation(
                checkpoint_path=checkpoint_path,
                render=False,         # Disable visualization
                max_steps=100,
                deterministic=True,
                save_frames=False,
                frame_interval=10,
                env_seed=686
            )
            
            print("\n✅ Evaluation completed (without visualization)!")
            print(f"   Final colony size: {results['final_num_cells']} cells")
            print(f"   Total reward: {results['total_reward']:.3f}")
            print(f"   Episode length: {results['num_steps']} steps")
            print("\n💡 To try visualization again:")
            print("   python evaluations/ppo_eval.py --checkpoint ppo_checkpoints/ppo_colony_final.pt")
            
        except Exception as e2:
            print(f"❌ Evaluation also failed: {e2}")
            print("💡 Check that the model and environment are compatible")


if __name__ == "__main__":
    main()