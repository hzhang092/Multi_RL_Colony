# Goal
Design and train multi-agent reinforcement learning policies for simulated bacterial cells on a spatial grid so that learned local behaviors produce realistic colony-level morphology and dynamics. Compare learned colonies to calibrated rule-based ABM outputs using the lab’s feature set (aspect ratio, anisotropy, density, convexity, Fourier descriptors, growth-rate deviations). Analyze how reward design (individual vs colony vs hybrid) drives emergence of cooperation, competition, and adaptive dormancy; and show that RL can produce colony features that match experimental distributions or exceed simple heuristics. Implement RL algorithms and training infrastructure in PyTorch, and publish code, visualizations, and an ACL-style report.

# Design (under construction)
## Environment
32x32 grid?

## Marcov Decision Process
- states: cell center (x,y), energy, age, alive_flag
- actions: 0 = stay, 1 = eat/grow, 2 = divide, 3 = produceEnzyme (optional), 4 = dormancy
- rewards:
    - individual rewards: nutrient_consumed, division_success, crowd_penalty, action_cost
    - shared (colony) rewards: colony_size, distance between simualated features and authetic features

### Dynamics per step:
- Eating: consumes nutrient at agent’s cell → +energy.
- Division: if energy ≥ threshold and adjacent empty cell exists → spawn child with half energy.
- Nutrient diffusion/decay/regeneration: simple per-step diffusion kernel + small stochastic refill.
- Death: if energy < threshold for many steps, agent dies (alive=False).

