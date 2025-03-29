from tqdm import tqdm
import numpy as np
import pandas as pd
import altair as alt
import os


def create_invert_map(l):
    return {v:i for i,v in enumerate(l)}

class BlackJack:
    #variables declaration
    cards = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    agent_sums = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    dealers_cards = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ace_usability = [False, True]  # Referenced as 0, 1
    actions_possible = ["S", "H"]  # Referenced as 0, 1

    agent_sums_map, dealers_cards_map, ace_usability_map, actions_map, cards_map = map(
        create_invert_map,
        [agent_sums, dealers_cards, ace_usability, actions_possible, cards],
    )

    def __init__(self, M, epsilon, alpha, seed=0):
        self.seed = seed
        self.M = M
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q_hist = None
        self.initialize()

    def initialize(self):
        #define q
        self.Q = np.zeros(len(self.agent_sums), len(self.dealers_cards), len(self.ace_usability), len(self.actions_possible))
        self.C = np.zeros_like(self.Q, dtype=int) # for counting
        if not os.path.exists('data') :
            os.mkdir("data")

    def sample_cards(self, N):
        #Draws N samples for the infinite 13 card deck.
        sample_cards = np.random.choice(self.cards, size=N, replace=True)
        return [c if c=="A" else int(c) for c in sample_cards]
    
    def map_to_indices(self, agent_sum, dealers_card, useable_ace):
        #Map from agent_sum (12-21), dealers_card(ace-10) and useable_ace (True, False) to the indices of Q
        return (
            self.agent_sums_map[agent_sum],
            self.dealers_cards_map[dealers_card],
            self.ace_usability_map[useable_ace],
        )
    
    def behabior_policy(self, agent_sum, dealers_card, useable_ace):
        #Returns H (Hit) or S (Stick) to determine the actions to take during the game.
        agent_sum_idx, dealers_card_idx, useable_ace_idx = self.map_to_indices(agent_sum, dealers_card, useable_ace)
        greedy_action = self.Q[agent_sum_idx, dealers_card_idx, useable_ace_idx].argmax()

        do_greedy = np.random.binomial(1, 1 - self.epsilon + (self.epsilon / 2))
        if do_greedy:
            return self.actions_possible[int(greedy_action)]
        else:
            return self.actions_possible[int(not greedy_action)]
    
    def target_policy(self, agent_sum, dealers_card, useable_ace):
        #Now it's on-policy learning so it's the sam as behavior policy
        return self.behavior_policy(
            agent_sum=agent_sum, dealers_card=dealers_card, useable_ace=useable_ace
        )

    def is_ratio(self, states_remaining, actions_remaining):
        #The Importance Sampling ratio that can be overwritten for off policy control.
        return 1
    
    @staticmethod
    def calc_sum_useable_ace(cards):
        #Returns the sum-value of cards and whether there is a useable ace
        cards_sum = sum([c for c in cards if c != "A"])
        ace_count = len([c for c in cards if c == "A"])

        if ace_count == 0:
            return cards_sum, False
        else :
            cards_sum_0 = cards_sum + ace_count
            cards_sum_1 = cards_sum + 10 + ace_count

            if cards_sum_1 > 21 :
                return cards_sum_0, False
            return cards_sum_1, True
    

    def play_game(self):

        agent_cards = self.sample_cards(2)
        dealers_card = self.sample_cards(1)[0]

        states = [[agent_cards, dealers_card]]
        actions = []

        #hit until agent_sum >=12
        while True:
            agent_cards, dealers_card = states[-1]
            agent_sum, _ = self.calc_sum_useable_ace(agent_cards)
            if agent_sum < 12 :
                actions.append("H")
                agent_cards_next = agent_cards + self.sample_cards(1)
                states.append([agent_cards_next, dealers_card])
            else: 
                break

        # Play game according to agents policy
        while True:
            agent_cards, dealers_card = states[-1]
            agent_sum , useable_ace = self.calc_sum_useable_ace(agent_cards)

            action = self.behabior_policy(agent_sum, dealers_card, useable_ace)
            actions.append(action)

            if action == "S":
                states.append([agent_cards, dealers_card])
                break
            else :
                agent_cards_next = agent_cards + self.sample_cards(1)
                states.append([agent_cards_next, dealers_card])

                agent_sum_next, useable_ace = self.calc_sum_useable_ace(
                    agent_cards_next
                )

                if agent_sum_next > 21:
                    how = "bust"
                    reward = -1
                    return states, actions, reward, how
        #Dealer play
        dealers_cards = states[-1][1] + self.sample_cards(1)
        while True :
            dealers_sum, _ = self.calc_sum_useable_ace(dealers_card)
            if dealers_sum > 21:
                how = "dealers bust : " + ",".join([str(c) for c in dealers_cards]) 
                reward = 1
                return states, actions, reward, how
            if dealers_sum > 16:
                break
            else: 
                dealers_cards += self.sample_cards(1)

        agent_sum, useable_ace = self.calc_sum_useable_ace(states[-1][0])

        if agent_sum == dealers_sum:
            return states, actions, 0, f"{agent_sum} = {dealers_sum}"
        elif agent_sum > dealers_sum:
            return states, actions, 1, f"{agent_sum} > {dealers_sum}"
        else:
            return states, actions, -1, f"{agent_sum} < {dealers_sum}"

    def get_hyper_str(self):
        #Returns a string uniquely identifying the class arguments.
        return f"M{self.M}__epsilon{str(self.epsilon).replace('.', '_')}__alpha{str(self.alpha).replace('.', '_')}__seed{str(self.seed)}"

    def get_file_names(self):
        hyper_str = self.get_hyper_str()
        Q_name = "data/Q_" + hyper_str
        C_name = "data/C_" + hyper_str
        Q_hist_name = "data/Q_hist_" + hyper_str
        return Q_name, C_name, Q_hist_name

    def save(self):
        Q_name, C_name, Q_hist_name = self.get_file_names()
        print(f"Saving {Q_name}")
        np.save(Q_name, self.Q)
        print(f"Saving {C_name}")
        np.save(C_name, self.C)
        if self.Q_hist is not None:
            print(f"Saving {Q_hist_name}")
            self.Q_hist.to_pickle(Q_hist_name + ".pkl")

    def load(self):
        Q_name, C_name, Q_hist_name = self.get_file_names()
        print(f"Loading...")
        self.Q = np.load(Q_name + ".npy")
        self.C = np.load(C_name + ".npy")
        try:
            self.Q_hist = pd.read_pickle(Q_hist_name + ".pkl")
        except:
            print("No Q hist to load")

    def _get_ms(self):
        #Returns a list of episodes index's (sometimes called 'm's) for which we record action-value pairs.
        return list(range(0, self.M + 1, 1000))

