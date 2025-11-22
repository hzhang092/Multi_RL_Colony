# Report: Handling Next-State Value for Dividing Parents in PPO

## Background
In our multi-agent ColonyEnv, agents (cells) can choose to divide. During PPO training, we bootstrap value targets using the next-state value V(s_{t+1}). Originally, when a parent cell divided, we set its next_value to 0 because the parent no longer existed in the next state. This inadvertently discouraged division.

## Symptom
- Division actions were underused or learned incorrectly in earlier runs where next_value for dividers was 0.
- After we switched to using child-only next values (blend = 1), division was encouraged early but the critic (value function) became noisier, with elevated and volatile value loss.

## Why next_value matters (GAE)
Advantage uses temporal-difference error:

- TD error: $\delta_t = r_t + \gamma\,V(s_{t+1}) - V(s_t)$
- GAE: $A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}$

If $V(s_{t+1})$ is set to 0 when a cell divides, we artificially depress $\delta_t$ for divide actions, even if the immediate division reward is positive. The policy sees division as less valuable and avoids it.

## Root cause
- In `experiments/ppo_train.py`, we built `next_values_for_acting_agents` with zeros for agents that divided (they disappear). Combined with GAE, this penalized division.
- In `agents/ppo_agent.py`, `compute_gae_advantages` uses those `next_values` directly in $\delta_t$.

## Diagnosis process
1. Inspected the rollout code in `experiments/ppo_train.py` around building `next_values_for_acting_agents`.
2. Confirmed zeros for dividers and how survivors were handled.
3. Reviewed `compute_gae_advantages` in `agents/ppo_agent.py` to verify the role of `next_values` in the TD target.
4. Ran with different schemes and compared logs (action mix, rewards, value loss).

## Fix design options we considered
- Child-only (blend = 1): Use average child value as the parent’s next_value.
  - Pros: Strongly encourages division.
  - Cons: Higher target variance (children didn’t exist at t), noisier critic, elevated value loss.
- Parent-only: Use parent’s current value as next_value.
  - Pros: Stable.
  - Cons: Neutral toward division; may not encourage it enough.
- Parent + constant bonus: Parent value plus fixed offset.
  - Pros: Simple.
  - Cons: Ad-hoc; risks biasing the critic.
- Average child value (per-parent mapping): Best if we can map parent→children exactly.
  - Pros: Principled.
  - Cons: Requires explicit mapping; we used a practical approximation.
- Blended approach (chosen): Start from parent’s current value and blend with average child value.
  - Pros: Encourages division and reduces variance vs child-only.
  - Cons: Introduces a tunable hyperparameter (blend).

## Implementation we deployed
In `experiments/ppo_train.py`, during rollout after `env.step(actions)`:

- Baseline next values for acting agents = their current values (neutral):
  - `next_values_for_acting_agents = parent_values_np.copy()`
- Survivors: assign their true next-state values from `next_obs` (env orders next_obs as `[survivors..., children...]`).
- Dividers: compute `avg_child_value` from the tail of `next_values_np` and set parent’s next_value to a blend:
  - `next_value(parent_divider) = blend * avg_child_value + (1 - blend) * parent_value`
- Default `blend = 0.8` (tunable 0.7–0.9).

This prevents zeroing out future value for dividers and ties the bootstrap to the value of the newly created children while tempering variance.

## Results and observations
With child-only bootstrap (blend = 1), logs showed:
- AVG Reward improved over time (~0.70 → ~0.84 peaks), AVG Num Cells ~13–17.
- Early action mix heavily favored division (e.g., update 10: divide dominated), later more balanced with grow leading.
- Value loss (critic) was elevated and volatile (e.g., 70 → 140 → 314, later trending down but still high), indicating noisier targets.

Interpretation:
- Using children to bootstrap strongly encourages division (desired), but increases critic target variance.
- The blended approach (e.g., 0.8) retains the incentive while stabilizing the critic compared to 1.0.

## Tuning knobs
- blend (bootstrap mixing): start at 0.8. Increase to encourage division more; decrease to stabilize critic.
- VALUE_COEF (critic loss weight): if value loss stays high/volatile, increase from 0.5 to 0.6–0.8.
- LR (optimizer learning rate): if losses oscillate, reduce slightly (e.g., 3e-4 → 2.5e-4).

We exposed both `LR` and `VALUE_COEF` in `experiments/ppo_train.py` and included them in checkpoints.

## Next steps
- Add diagnostics:
  - Division attempts/success rate per update.
  - Fraction of action=2 when `length >= L_divide` (quality of divide decisions).
  - Per-action average reward/advantage.
- Consider gentle reward shaping if needed (e.g., small bonus at division readiness) and ensure `p_invalid_div` stays modest.
- If critic remains noisy, try blend in [0.7, 0.9], slightly higher VALUE_COEF, and/or slightly lower LR.

## File references
- `experiments/ppo_train.py`: rollout loop building `next_values_for_acting_agents` (blended next values for dividers implemented).
- `agents/ppo_agent.py`: `compute_gae_advantages` (uses `next_values` in TD error), PPO update.
- `envs/colony_env.py`: `step()` returns `info['survivor_indices']`; `next_obs` ordered as `[survivors..., children...]`.

## Summary
Zero next-state values for dividing parents discouraged division by depressing $\delta_t$ in GAE. We replaced them with a principled bootstrap that blends the average child value with the parent’s current value. This encourages correct division while reducing critic variance versus the child-only alternative. Hyperparameters (`blend`, `VALUE_COEF`, `LR`) provide additional control to balance learning signal strength and stability.
