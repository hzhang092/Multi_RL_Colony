# Agent Evaluation and Visualization

This directory contains scripts for evaluating trained PPO agents and visualizing colony growth dynamics.

## Files

### `ppo_eval.py` - Comprehensive Evaluation Script
The main evaluation script with full configurability and command-line interface.

**Features:**
- Load trained agents from checkpoint files
- Real-time colony growth visualization
- Deterministic or stochastic action evaluation
- Performance metrics tracking
- Frame saving for creating videos
- Configurable evaluation parameters

**Usage:**
```bash
# Basic usage - automatically finds latest checkpoint
python evaluations/ppo_eval.py

# Specify a particular checkpoint
python evaluations/ppo_eval.py --checkpoint ppo_checkpoints/ppo_colony_500.pt

# Run without visualization (metrics only)
python evaluations/ppo_eval.py --no_render --max_steps 100

# Save frames for video creation
python evaluations/ppo_eval.py --save_frames --frame_interval 2

# Use stochastic actions (as during training)
python evaluations/ppo_eval.py --stochastic

# Full configuration example
python evaluations/ppo_eval.py \
    --checkpoint ppo_checkpoints/ppo_colony_final.pt \
    --max_steps 500 \
    --save_frames \
    --frame_interval 5 \
    --seed 42
```

**Command-line Arguments:**
- `--checkpoint, -c`: Path to checkpoint file (auto-detects if not provided)
- `--max_steps, -s`: Maximum evaluation steps (default: 300)
- `--no_render`: Disable visualization
- `--stochastic`: Use stochastic actions instead of deterministic
- `--save_frames`: Save visualization frames to disk
- `--frame_interval`: Steps between frames (default: 5)
- `--seed`: Random seed for environment (default: 686)

### `visualize_agent.py` - Simple Visualization Script
A simplified wrapper for quick colony visualization without command-line arguments.

**Features:**
- Automatically finds latest checkpoint
- One-click colony visualization
- No configuration needed
- Perfect for quick testing

**Usage:**
```bash
# Just run it - no arguments needed!
python visualize_agent.py
```

## Visualization Output

The visualization shows:
- **Real-time colony growth**: Watch cells grow, divide, and interact
- **Step information**: Current step number and metrics
- **Cell count**: Number of active cells in the colony
- **Rewards**: Step-wise and cumulative reward values
- **Colony morphology**: Visual representation of colony shape evolution

## Performance Metrics

The evaluation scripts track and report:
- **Total reward**: Cumulative reward over the episode
- **Average reward per step**: Mean reward across all timesteps
- **Final cell count**: Number of cells at episode end
- **Maximum cells reached**: Peak colony size during evaluation
- **Episode length**: Number of steps taken
- **Termination reason**: Whether episode ended naturally or was truncated

## Checkpoint File Formats

The scripts support two checkpoint formats:

1. **Full checkpoints** (recommended):
   ```python
   {
       'policy_state_dict': ...,
       'optimizer_state_dict': ...,
       'update': training_step,
       'total_steps': environment_steps,
       'best_reward': best_performance,
       'hyperparameters': {...}
   }
   ```

2. **State dict only**:
   ```python
   # Just the policy parameters
   torch.save(agent.policy.state_dict(), 'model.pt')
   ```

## Creating Training Videos

To create videos of colony growth:

1. **Save frames during evaluation:**
   ```bash
   python evaluations
/ppo_eval.py --save_frames --frame_interval 2 --max_steps 500
   ```

2. **Convert frames to video using ffmpeg:**
   ```bash
   ffmpeg -r 10 -i evaluation_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p colony_growth.mp4
   ```

## Troubleshooting

**No checkpoint files found:**
- Make sure you've trained a model first: `python evaluations/ppo_train.py`
- Check that checkpoint files exist in `ppo_checkpoints/` directory

**Visualization not showing:**
- Ensure matplotlib backend supports GUI: `import matplotlib; matplotlib.use('TkAgg')`
- Try running with `--no_render` to test without visualization

**CUDA/GPU issues:**
- The scripts automatically handle CPU/GPU detection
- Models trained on GPU can be evaluated on CPU and vice versa

**Performance issues:**
- Reduce `--frame_interval` to update visualization less frequently
- Use `--no_render` for faster metric-only evaluation
- Reduce `--max_steps` for shorter evaluation runs

## Examples

**Quick start after training:**
```bash
python visualize_agent.py
```

**Detailed evaluation with metrics:**
```bash
python evaluations/ppo_eval.py --max_steps 1000 --checkpoint ppo_checkpoints/ppo_colony_final.pt
```

**Create training video:**
```bash
python evaluations/ppo_eval.py --save_frames --max_steps 300 --frame_interval 3
ffmpeg -r 8 -i evaluation_frames/frame_%04d.png -c:v libx264 colony_training_result.mp4
```