import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class Bandit:
    # k-armed bandit
    def __init__(
        self,
        k=10,
        init_true_values=None,
        init_q_values=None,
        eps=None,
        alpha=None,
        ucb=False,
        c=None,
    ):
        self.k = k
        self.init_true_values = init_true_values
        self.init_q_values = init_q_values
        self.alpha = alpha
        self.ucb = ucb
        self.c = c
        # not none eps mean e-gready algorithm
        self.is_greedy = True if eps is None else False
        self.eps = eps
        self.action_count = np.zeros(self.k)
        self.n = 0

        self.true_values = (
            np.copy(self.init_true_values)
            if self.init_true_values is not None
            else np.random.randn(self.k)
        )

        self.best_arm_index = np.argmax(self.true_values)

        self.q_values = (
            np.copy(self.init_q_values) if self.init_q_values is not None else np.zeros(self.k)
        )

    def reset(self):
        self.n = 0
        self.action_count = np.zeros(self.k)

        self.true_values = (
            np.copy(self.init_true_values)
            if self.init_true_values is not None
            else np.random.randn(self.k)
        )

        self.best_arm_index = np.argmax(self.true_values)

        self.q_values = (
            np.copy(self.init_q_values) if self.init_q_values is not None else np.zeros(self.k)
        )

    def pull_arm(self, arm_index):
        revard = np.random.normal(self.true_values[arm_index], 1.0)
        return revard

    def train(self):
        self.n += 1

        if self.ucb:
            ucb_estimates = self.q_values + self.c * np.sqrt(
                np.log(self.n) / (self.action_count + 1e-5)
            )
            arm_index = np.random.choice(np.where(ucb_estimates == np.max(ucb_estimates))[0])
        else:
            eps = np.random.rand()
            arm_index = np.random.choice(np.where(self.q_values == np.max(self.q_values))[0])

            if not self.is_greedy and eps <= self.eps:
                arm_index = np.random.choice(self.k, 1).item()

        revard = self.pull_arm(arm_index)
        self.action_count[arm_index] += 1
        step_size = 1.0 / self.action_count[arm_index] if self.alpha is None else self.alpha

        self.q_values[arm_index] = self.q_values[arm_index] + step_size * (
            revard - self.q_values[arm_index]
        )

        is_best_arm_selected = arm_index == self.best_arm_index

        return revard, is_best_arm_selected


def simulate(bandits, runs=2000, time_steps=1000):
    revards = np.zeros((len(bandits), runs, time_steps))
    best_actions = np.zeros((len(bandits), runs, time_steps))
    for i, bandit in enumerate(bandits):
        for run in tqdm(range(runs)):
            bandit.reset()
            for t in range(time_steps):
                revard, is_best = bandit.train()
                revards[i, run, t] = revard
                best_actions[i, run, t] = int(is_best)

    return revards, best_actions


def experiment_1():
    bandits = [Bandit(10), Bandit(10, eps=0.1), Bandit(10, eps=0.01)]

    revards, best_actions = simulate(bandits)

    average_revard = np.mean(revards, axis=1)
    optimal_action_perc = np.mean(best_actions, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 14))

    ax1.plot(average_revard[0], label="greedy")
    ax1.plot(average_revard[1], label="eps-greedy, eps=0.1")
    ax1.plot(average_revard[2], label="eps-greedy, eps=0.01")
    ax1.legend()
    ax1.set_xlabel("time steps")
    ax1.set_ylabel("average revards")

    ax2.plot(optimal_action_perc[0], label="greedy")
    ax2.plot(optimal_action_perc[1], label="eps-greedy, eps=0.1")
    ax2.plot(optimal_action_perc[2], label="eps-greedy, eps=0.01")
    ax2.legend()
    ax2.set_xlabel("time steps")
    ax2.set_ylabel("optimal action percent")

    plt.savefig("plots/experiment_1.jpg")


def experiment_2():
    bandits = [Bandit(10, init_q_values=np.ones(10) * 5, alpha=0.1), Bandit(10, eps=0.1)]

    revards, best_actions = simulate(bandits)

    average_revard = np.mean(revards, axis=1)
    optimal_action_perc = np.mean(best_actions, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 14))

    ax1.plot(average_revard[0], label="Optimistic (q=5), greedy")
    ax1.plot(average_revard[1], label="eps-greedy, eps=0.1")
    ax1.legend()
    ax1.set_xlabel("time steps")
    ax1.set_ylabel("average revards")

    ax2.plot(optimal_action_perc[0], label="Optimistic (q=5), greedy")
    ax2.plot(optimal_action_perc[1], label="eps-greedy, eps=0.1")
    ax2.legend()
    ax2.set_xlabel("time steps")
    ax2.set_ylabel("optimal action percent")

    plt.savefig("plots/experiment_2.jpg")


def experiment_3():
    bandits = [Bandit(10, ucb=True, c=2), Bandit(10, eps=0.1)]

    revards, best_actions = simulate(bandits)

    average_revard = np.mean(revards, axis=1)
    optimal_action_perc = np.mean(best_actions, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 14))

    ax1.plot(average_revard[0], label="UCB c=2")
    ax1.plot(average_revard[1], label="eps-greedy, eps=0.1")
    ax1.legend()
    ax1.set_xlabel("time steps")
    ax1.set_ylabel("average revards")

    ax2.plot(optimal_action_perc[0], label="UCB c=2")
    ax2.plot(optimal_action_perc[1], label="eps-greedy, eps=0.1")
    ax2.legend()
    ax2.set_xlabel("time steps")
    ax2.set_ylabel("optimal action percent")

    plt.savefig("plots/experiment_3.jpg")


if __name__ == "__main__":
    # experiment_1()
    experiment_3()
