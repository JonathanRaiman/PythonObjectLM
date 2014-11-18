import numpy as np

from utils import norm_pdf, entropy

class HierarchicalObservation:
    def __init__(self, covariance):
        self.covariance = covariance
    
    def observation_vector(self, positive=[], negative = []):
        """
        Take labels as observations being positive or negative
        and find the indices associated, along with a vector
        for those observations in the tree space
        
        Inputs
        ------
        
        positive list: list of labels that are positive
        negative list: list of labels that are negative
        
        Outputs
        -------
        tuple : positive feature indices, then negative feature
                indices as single list, along with an observation
                vector with +0.5 for positive and -0.5 for negative
                in the correct order for the observation.
        
        """
        return (positive + negative, np.concatenate([
            np.ones(len(positive)) * 0.5, 
            np.ones(len(negative)) * -0.5]))
    
    def _calculate_conditional_probabilities(self, y, indices, remaining_indices, singular=False):
        assert len(y) == len(indices), "Not the same observation length as indices"
        probs = np.zeros([len(remaining_indices), 2])
        if singular:
          diagonal_noise = np.eye(len(indices) + 1) * 1e-6
          for k, unseen_index in enumerate(remaining_indices):
              probs[k, 0] = norm_pdf(np.concatenate([y, [0.5]]),
                       self.covariance[indices + [unseen_index], :][:, indices + [unseen_index]] + diagonal_noise)
              probs[k, 1] = norm_pdf(np.concatenate([y, [-0.5]]),
                       self.covariance[indices + [unseen_index], :][:, indices + [unseen_index]] + diagonal_noise)
        else:
          for k, unseen_index in enumerate(remaining_indices):
              probs[k, 0] = norm_pdf(np.concatenate([y, [0.5]]),
                       self.covariance[indices + [unseen_index], :][:, indices + [unseen_index]])
              probs[k, 1] = norm_pdf(np.concatenate([y, [-0.5]]),
                       self.covariance[indices + [unseen_index], :][:, indices + [unseen_index]])
        # marginalize probabilities

        return (probs.T / probs.sum(axis=1)).T

    def _calculate_conditional_probabilities_jointly(self, y, indices, remaining_indices):
        assert len(y) == len(indices), "Not the same observation length as indices"
        probs = np.zeros([len(remaining_indices), 2])
        remaining_indices_set = set(remaining_indices)
        for k, unseen_index in enumerate(remaining_indices):
            cov_slice = self.covariance[indices + [unseen_index] + list(remaining_indices_set - set([unseen_index])), :][:, indices + [unseen_index] + list(remaining_indices_set - set([unseen_index]))]
            probs[k, 0] = norm_pdf(np.concatenate([y, [0.5] + [-0.5] * (len(remaining_indices) -1)]), cov_slice)
            probs[k, 1] = norm_pdf(np.concatenate([y, [-0.5] * len(remaining_indices)]), cov_slice)
        # marginalize probabilities

        return (probs.T / probs.sum(axis=1)).T

    def conditional_probabilities_jointly(self,
                                  positive = [], 
                                  negative = [], 
                                  remaining_indices = [],
                                  sigma = None,
                                  num_steps = 50,
                                  num_trials = 100,
                                  approximate = True):
        """
        Take labels as observations being positive or negative
        and find the conditional probability for each
        unassigned variable in the tree given the covariance matrix.
        
        Inputs
        ------
        
        positive list: list of labels that are positive
        negative list: list of labels that are negative
        sigma float : a prior variance on tree vertices
        num_steps int : how many steps in random walk for
                        Monte Carlo covariance technique
        num_trials int : how many samples of random walks to
                         use for each dimension in covariance
        approximate boolean : whether to use matrix inversion
                              or Monte Carlo to get covariance
                              matrix.
                              
        Outputs
        -------
        
        tuple <list<str> labels, np.array[n,2] probs> : return
            the labels of unassigned variables and their
            conditional probabilities
            
        """
        indices, y = self.observation_vector(positive,negative)
        probs = self._calculate_conditional_probabilities_jointly(y, indices, remaining_indices)
        
        return (remaining_indices, probs)
    
    def conditional_probabilities(self,
                                  positive = [], 
                                  negative = [], 
                                  remaining_indices = [],
                                  sigma = None,
                                  num_steps = 50,
                                  num_trials = 100,
                                  singular = False,
                                  approximate = True):
        """
        Take labels as observations being positive or negative
        and find the conditional probability for each
        unassigned variable in the tree given the covariance matrix.
        
        Inputs
        ------
        
        positive list: list of labels that are positive
        negative list: list of labels that are negative
        sigma float : a prior variance on tree vertices
        num_steps int : how many steps in random walk for
                        Monte Carlo covariance technique
        num_trials int : how many samples of random walks to
                         use for each dimension in covariance
        approximate boolean : whether to use matrix inversion
                              or Monte Carlo to get covariance
                              matrix.
                              
        Outputs
        -------
        
        tuple <list<str> labels, np.array[n,2] probs> : return
            the labels of unassigned variables and their
            conditional probabilities
            
        """
        indices, y = self.observation_vector(positive,negative)
        probs = self._calculate_conditional_probabilities(y, indices, remaining_indices, singular)
        
        return (remaining_indices, probs)
    
    def posterior_entropy(self,
                                  positive = [], 
                                  negative = [], 
                                  remaining_indices = []):
        
        indices, y = self.observation_vector(positive,negative)
        entropies = np.zeros([len(remaining_indices)])
        remaining_indices_set = set(remaining_indices)

        for k, unseen_index in enumerate(remaining_indices):
            # suppose we observe this element. What is the new entropy ?
            
            test_indices = indices + [unseen_index]
            
            prob_pos = norm_pdf(np.concatenate([y, [0.5]]),
                     self.covariance[test_indices, :][:, test_indices])
            prob_neg = norm_pdf(np.concatenate([y, [-1.0]]),
                     self.covariance[test_indices, :][:, test_indices])
            
            # marginalize:
            Z = (prob_pos + prob_neg)
            prob_pos /= Z
            prob_neg /= Z
            
            # get the entropy of remaining elements:
            post_indices = remaining_indices_set.copy()
            post_indices.remove(unseen_index)
            post_probs_pos = self._calculate_conditional_probabilities(np.concatenate([y , [0.5]]),
                                                                       test_indices,
                                                                       post_indices,
                                                                       self.covariance)
            post_probs_neg = self._calculate_conditional_probabilities(np.concatenate([y , [-.5]]),
                                                                       test_indices,
                                                                       post_indices,
                                                                       self.covariance)
            # expected entropy:
            entropies[k] = (
                prob_pos * entropy(post_probs_pos) +
                prob_neg * entropy(post_probs_neg)
            )
        return (remaining_indices, entropies)