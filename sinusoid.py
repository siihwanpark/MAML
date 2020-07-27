# Code is almost same with torchmeta
# https://github.com/tristandeleu/pytorch-meta

import numpy as np
import torch
from torch.utils.data import Dataset

class Sinusoid(Dataset):
    """
    Simple regression task, based on sinusoids, as introduced in [1].
    Parameters
    ----------
    k_shot : int
        Number of shots per task.
    q_query : int
        Number of querys per task.
    num_tasks : int (default: 1,000,000)
        Overall number of tasks to sample.
    Notes
    -----
    The tasks are created randomly as random sinusoid function. The amplitude
    varies within [0.1, 5.0], the phase within [0, pi], and the inputs are
    sampled uniformly in [-5.0, 5.0]. Due to the way PyTorch handles datasets,
    the number of tasks to be sampled needs to be fixed ahead of time (with
    `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, k_shot, q_query=15, num_tasks=1000000):
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks

        # Set ranges of inputs, amplitude, and phase
        self._input_range = np.array([-5.0, 5.0])
        amplitude_range = np.array([0.1, 5.0])
        phase_range = np.array([0., np.pi])

        # Sample amplitudes and phases
        self._amplitudes = np.random.uniform(amplitude_range[0],
            amplitude_range[1], size=self.num_tasks)
        self._phases = np.random.uniform(phase_range[0], phase_range[1],
            size=self.num_tasks)

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        # Output: Support_input, Support_target, Query_input, Query_target
        # Shape: (K, 1), (K, 1), (Q, 1), (Q, 1)
        # where K:shots, Q:querys
        amplitude, phase= self._amplitudes[index], self._phases[index]
        
        # SinusoidTask
        inputs = np.random.uniform(self._input_range[0], self._input_range[1],
            size=(self.k_shot + self.q_query, 1))
        targets = amplitude * np.sin(inputs + phase)

        # Output tensor
        inputs = torch.tensor(inputs, dtype = torch.float)
        targets = torch.tensor(targets, dtype = torch.float)

        return inputs[:self.k_shot], targets[:self.k_shot], inputs[self.k_shot:], targets[self.k_shot:]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset = Sinusoid(k_shot=10, q_query=15, num_tasks=2000000)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=True)
    # Shape: (M, K, 1), (M, K, 1), (M, Q, 1), (M, Q, 1) where M:batch_size
    train_inputs, train_targets, test_inputs, test_targets = next(iter(train_loader))
    
    print(train_inputs.shape)       # Shape: (10, 10, 1)
    print(train_targets.shape)      # Shape: (10, 10, 1)
    print(test_inputs.shape)        # Shape: (10, 15, 1)
    print(test_targets.shape)       # Shape: (10, 15, 1)