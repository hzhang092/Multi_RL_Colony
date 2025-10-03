# policy gradient agent with manual policy representation, using REINFORCE algorithm
import numpy as np

class manualPolicyGradient:
    def __init__(self):
        self.theta = {}
        for n in [0,1,2]:
            for p in [0,1,2]:
                self.theta[(n,p)] = [0.0, 0.0, 0.0]  # logits, not probabilities
                
    def get_action_probs(self, state):
        """Convert logits to probabilities using softmax."""
        logits = self.theta[state]
        exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def choose_action(self, state):
        """ sample action based on probabilities """
        probs = self.get_action_probs(state)
        return np.random.choice(len(probs), p=probs), probs

    def update(self, episode_states, episode_actions, episode_rewards, learning_rate):
        """ Update policy parameters using REINFORCE algorithm """
        G = 0 # return (discounted sum of rewards)
        gamma = 0.9 # discount factor
        
        # calculate returns
        returns = []
        
        for t in reversed(range(len(episode_rewards))):
            """
            We use reversed order because in REINFORCE, the return G at each time step t is the sum of discounted rewards from t onward.
            By iterating backwards, we efficiently compute G for each step without recomputing sums.
            """
            G = episode_rewards[t] + gamma * G  # compute discounted return
            returns.insert(0, G)  # insert at the beginning to maintain order
            
        # normalize returns for stability
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # update policy parameters
        for t in range(len(episode_rewards)):
            state = episode_states[t]
            action = episode_actions[t]
            probs = self.get_action_probs(state)
            
            # calculate gradient
            gradient = -probs.copy()  # ∂log(π)/∂θ = (1 - π) for chosen action, -π for others
            gradient[action] += 1  # for chosen action

            # update parameters: theta = theta + alpha * G * gradient
            self.theta[state] += learning_rate * returns[t] * gradient