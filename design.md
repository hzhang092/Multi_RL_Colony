# Goal
Design and train multi-agent reinforcement learning policies for simulated bacterial cells on a spatial grid so that learned local behaviors produce realistic colony-level morphology and dynamics. Compare learned colonies to calibrated rule-based ABM outputs using the lab’s feature set (aspect ratio, anisotropy, density, convexity, Fourier descriptors, growth-rate deviations). Analyze how reward design (individual vs colony vs hybrid) drives emergence of cooperation, competition, and adaptive dormancy; and show that RL can produce colony features that match experimental distributions or exceed simple heuristics. Implement RL algorithms and training infrastructure in PyTorch, and publish code, visualizations, and an ACL-style report.

# Design (under construction)
## Environment
### reduced biophysics
- squared grid 64x64, each grid is either a cell or nutrient(empty)
- The colony starts with one cell, taking only 1 grid. Each cell can take more than one grid after they grow. When they take on more than 1 grid, the cell tends to divide with some probability. 
- when the cell takes only 1 grid, they can choose to grow vertically or diagonally
- when a cell decide grow, it 

## Marcov Decision Process
- states: 
    - cell center (x,y), 
    - cell orientation,
    - pole length
    - energy?

- actions: 
    - 0 = grow, 
    - 1 = divide, 
    - 2 = dormancy

- rewards:
    - individual rewards: nutrient_consumed, division_success, crowd_penalty, action_cost
    - shared (colony) rewards: colony_size, distance between simualated features and authetic features

### Dynamics per step:
- Eating: consumes nutrient at agent’s cell → +energy.
- Division: if energy ≥ threshold and adjacent empty cell exists → spawn child with half energy.
- Nutrient diffusion/decay/regeneration: simple per-step diffusion kernel + small stochastic refill.
- Death: if energy < threshold for many steps, agent dies (alive=False).

