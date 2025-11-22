# 1. biophysics
- need to simulate real bacteria behaviour
- need to be variable

# 2. obvservation and award design
- observation
    - local view vs globa view
- award
    - division award
        - do not remove the parent cell, remove it next step. count it's award this step.
        - inherate award from parent cell
        - immediately award the cell for division Y

    - log avg award,?

#### "Number of agents can change between timesteps due to reproduction" how to handle this during training?
- rewards_batch = rewards_arr[:len(obs_batch)].tolist()


### division award
division award not able to assigned, full detail see "miscellaneous\divide_award_report.md"
- potential solutions:
    - do not remove the parent cell, remove it next step. count it's award this step.
    - inherate award from parent cell
    - immediately award the cell for division ✅
- solution used:
    colony_env.py: immediately award the cell for division, return awards for cells taken action, 
    ppo_train.py: fixed misaligned lists given to buffer

#### follow up issue
in ppo_train.py, next step values for divided cells are set to 0 since they do not exist in the next step. This turns out to discourage division in GAE algorithm 