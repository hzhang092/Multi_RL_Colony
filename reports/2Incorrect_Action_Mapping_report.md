# Report: Diagnosing and Resolving Incorrect Action Mapping in PPO Agent Training

## Problem Statement
During the training of a Proximal Policy Optimization (PPO) agent in a multi-agent Colony Environment (ColonyEnv), the agent exhibited unintended behavior. Specifically, the agent avoided division actions, which are critical for colony growth, and instead relied on maintaining large colonies to achieve high cell counts. This behavior deviated from the intended training dynamics and hindered the agent's ability to learn proper action strategies.

Initially, the issue was suspected to be an incorrect action mapping, where the agent's policy network might have been producing invalid or unintended actions. However, further investigation revealed that the root cause was a logic flaw in the training process rather than a mapping error.

## Core Cause Analysis
The root causes of the issue were identified through a systematic debugging process:

1. **Reward Structure:**
   - Observation: Training logs showed that the agent rarely selected the "divide" action, even when it was the optimal choice.
   - Analysis: The reward structure penalized invalid division actions too heavily. For example, if a cell attempted to divide without sufficient resources, the penalty was disproportionately high (-0.5) compared to the rewards for successful actions (+1). This discouraged the agent from exploring division actions.

2. **Environment Dynamics:**
   - Observation: The agent's performance improved in terms of total cell count, but the action distribution remained skewed towards "dormant" and "grow" actions.
   - Analysis: The environment was not reset at the start of each training update. This led to stale states where the agent exploited existing colonies rather than learning to grow new ones. For instance, colonies that were already large at the start of an update allowed the agent to achieve high rewards without dividing.

3. **Debugging Process:**
   - Step 1: Reviewed the action sampling logic in `ppo_train.py` and confirmed that the policy network outputs matched the intended action space.
   - Step 2: Analyzed the reward computation in `colony_env.py` and identified the disproportionate penalty for invalid division actions.
   - Step 3: Examined the training loop and discovered the lack of environment resets, which reinforced suboptimal behaviors.

These factors combined to create a training dynamic where the agent avoided division actions entirely, relying instead on other strategies to maximize rewards.

## Solution Implementation
To address the identified issues, the following fixes were implemented:

1. **Environment Resets:**
   - Implementation: The training loop in `ppo_train.py` was updated to reset the environment at the start of each training update.
   - Example: Before the fix, the environment state persisted across updates, allowing the agent to exploit existing colonies. After the fix, each update begins with a fresh environment state, promoting exploration and reducing the reinforcement of suboptimal behaviors.

2. **Reduced Penalty for Invalid Actions:**
   - Implementation: The reward structure in `colony_env.py` was modified to reduce the penalty for invalid division actions.
   - Example: Previously, an invalid division action incurred a penalty of -10. This was reduced to -2, making it less discouraging for the agent to explore division actions.

## Validation and Expected Outcomes
The implemented fixes are expected to:
- Promote balanced exploration of all actions, including division.
- Ensure proper training dynamics by starting each update from a fresh environment state.
- Lead to improved learning of action strategies, aligning the agent's behavior with the intended training objectives.

## Next Steps
1. Monitor training logs to ensure balanced action distributions and proper colony growth.
   - Example: Check if the frequency of "divide" actions increases over time.
2. Validate the effectiveness of the fixes through further analysis and testing.
   - Example: Compare the total rewards and action distributions before and after the fixes.
3. Refine the reward structure and training loop as needed based on training outcomes.