'''
Values: A - K (1 - 13)
Suites: Spades, Clubs, Diamonds, Hearts (0 - 3)
Sets: LSp, HSp, LCl, HCl, LDi, HDi, LHe, HHe (0 - 7)
Indexes: ASp, 2Sp, ... KSp, ACl, ... ADi ... AHe ... KHe (0 - 51)
'''
import random
import secrets
import numpy as np
import time
from utils import *

class NoMovesException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class Card:
    def __init__(self, suite, value):
        self.val = value
        self.sui = suite
        self.set = whatSet(suite, value)
        self.own = None

    def SV(self):
        return self.sui, self.val

    def __str__(self):
        return f"{card2str(self.sui, self.val)}"

class Deck:
    def __init__(self, players):
        self.players = players
        self.cards = []

        for s in range(0, 4):
            for v in range(1, 14):
                if v != 8:
                    self.cards.append(Card(s, v))

    def deal(self):
        random.shuffle(self.cards)

        while len(self.cards) > 0:
            for p in self.players:
                card = self.cards.pop()
                card.own = p
                p.add(card)

class Player:
    # strat 1 uses probability knowledge
    def __init__(self, id, buds, opps, strat: 1):
        self.id = id
        self.hand = [0] * 52
        self.sets = [0] * 8
        self.buds = buds
        self.opps = opps
        self.known_sets = []
        self.probabilities = []
        self.inds = {}
        self.strat = strat

        all = buds + opps
        print(all)

        for i, x in enumerate(all):
            self.probabilities.append([0] * 52)
            self.known_sets.append([0] * 8)
            self.inds[x] = i

    def __str__(self):
        return f"Player {self.id}"

    def printProbs(self):
        print(f"Probabilities in P{self.id}:")

        all = self.buds + self.opps

        for i, id in enumerate(all):
            if id in self.buds:
                mod = "* "
            else:
                mod = ""

            print(f"\t{mod}{id}: \t{[float(round(x, 1)) for x in self.probabilities[i]]}")

    def add(self, card):
        sui, val = card.SV()
        card_ind = card2ind(sui, val)
        self.hand[card_ind] = 1
        self.sets[whatSet(sui, val)] += 1

        for id, ind in self.inds.items():
            self.probabilities[ind][card_ind] = 0

    def initialProbBalance(self):
        n = len(self.probabilities)

        for i, c in enumerate(self.hand):
            sui, val = ind2card(i)

            if c == 1 or val == 8:
                for p in range(0, n):
                    self.probabilities[p][i] = 0
            else:
                prob = 1.0 / float(n)
                for p in range(0, n):
                    self.probabilities[p][i] = prob

    def callHeard(self, fromP, toP, sui, val, successful):
        from_ind = self.inds[fromP.id]
        to_ind = self.inds[toP.id]
        card_ind = card2ind(sui, val)
        card_set = whatSet(sui, val)

        self.known_sets[from_ind][card_set] = 1

        if successful:
            for id, ind in self.inds.items():
                if id == fromP.id:
                    self.probabilities[ind][card_ind] = 1
                else:
                    self.probabilities[ind][card_ind] = 0
        elif self.hand[card_ind] == 0:
            prev_prob = self.probabilities[from_ind][card_ind] + self.probabilities[to_ind][card_ind]

            self.probabilities[from_ind][card_ind] = 0
            self.probabilities[to_ind][card_ind] = 0

            other_probs = []

            for id, ind in self.inds.items():
                print(self.probabilities[ind][card_ind])
                if self.probabilities[ind][card_ind] != 0:
                    other_probs.append(ind)

            print(other_probs)

            additive = prev_prob / float(len(other_probs))

            for ind in other_probs:
                self.probabilities[ind][card_ind] += additive

    def checkAsk(self, fromP, suite, value):
        from_ind = self.inds[fromP.id]
        card_ind = card2ind(suite, value)
        card_set = whatSet(suite, value)

        self.known_sets[from_ind][card_set] = 1

        if self.hand[card_ind] == 1:
            self.hand[card_ind] = 0
            self.sets[card_set] -= 1
            fromP.add(Card(suite, value))

            self.probabilities[from_ind][card_ind] = 1

            return True

        prev_prob = self.probabilities[from_ind][card_ind]

        self.probabilities[from_ind][card_ind] = 0

        other_probs = []

        for id, ind in self.inds.items():
            print(self.probabilities[ind][card_ind])
            if self.probabilities[ind][card_ind] != 0:
                other_probs.append(ind)

        print(other_probs)

        additive = prev_prob / float(len(other_probs))

        for ind in other_probs:
            self.probabilities[ind][card_ind] += additive

        return False

    def processResponse(self, ask):
        card = card2ind(ask['suite'], ask['value'])
        n = len(self.probabilities)

        if ask['success']:
            for p in range(0, n):
                self.probabilities[p][card] = 0
        else:
            to_ind = self.inds[ask['to'].id]

            prev_prob = self.probabilities[to_ind][card]

            self.probabilities[to_ind][card] = 0

            other_probs = []

            for id, ind in self.inds.items():
                print(self.probabilities[ind][card])
                if self.probabilities[ind][card] != 0:
                    other_probs.append(ind)

            print(other_probs)

            additive = prev_prob / float(len(other_probs))

            for ind in other_probs:
                self.probabilities[ind][card] += additive

    def isValidAsk(self, card_ind):
        sui, val = ind2card(card_ind)
        set = whatSet(sui, val)
        return self.sets[set] != 0 and val != 8 and self.hand[card_ind] == 0

    def pickRandom(self):
        options = []

        for i, set in enumerate(self.sets):
            if set > 0:
                valid_set_list = [ind for ind in SET_SEG[i] if self.hand[ind] == 0]
                options.extend(valid_set_list)

        if len(options) == 0:
            raise NoMovesException('Player has no possible asks!')

        random.shuffle(options)

        to = secrets.choice(self.opps)
        to_ind = self.inds[to]
        card = options.pop()

        while self.probabilities[to_ind][card] == 0 and not self.isValidAsk(card):
            if len(options) == 0:
                raise NoMovesException('Player has no possible asks!')

            card = options.pop()

        return card, to, -1

    def pickAsk(self):
        if self.strat == 1:
            options_ids = self.opps
            options_cards = []
            options_prob = []

            possible = []

            for i, set in enumerate(self.sets):
                if set > 0:
                    valid_set_list = [ind for ind in SET_SEG[i] if self.hand[ind] == 0]
                    possible.extend(valid_set_list)

            if len(possible) == 0:
                raise NoMovesException('Player has no possible asks!')

            for id in options_ids:
                p_ind = self.inds[id]

                valid_probs = [self.probabilities[p_ind][c_ind] for c_ind in possible]

                best_ind = np.argmax(valid_probs)
                card = possible[best_ind]
                sui, val = ind2card(card)
                prob = self.probabilities[p_ind][card] + (self.known_sets[p_ind][whatSet(sui, val)] * 0.2)

                options_cards.append(card)
                options_prob.append(prob)

            options = list(zip(options_ids, options_cards, options_prob))
            options = sorted(options, key=lambda x: x[2])

            card = options[-1][1]

            while len(options) > 1 and not self.isValidAsk(card):
                del options[-1]
                card = options[-1][1]

            if not self.isValidAsk(card):
                card, to, prob = self.pickRandom()
            else:
                to = options[-1][0]
                prob = options[-1][2]
        else:
            card, to, prob = self.pickRandom()

        suite, value = ind2card(card)

        return {'to': to, 'suite': suite, 'value': value, 'prob': prob}

class Game:
    def __init__(self, players):
        self.ids = list(set(players))
        self.teamA = self.ids[::2]
        self.teamB = self.ids[1::2]
        self.players = []
        self.lastAsk = None
        self.winner = None

        for id in self.ids:
            if id in self.teamA:
                buds = self.teamA.copy()
                buds.remove(id)
                opps = self.teamB.copy()
                strat = 1                   # change to have one team use random strat
            else:
                buds = self.teamB.copy()
                buds.remove(id)
                opps = self.teamA.copy()
                strat = 1

            self.players.append(Player(id, buds, opps, strat))

        self.deck = Deck(self.players)
        self.deck.deal()

        for p in self.players:
            p.initialProbBalance()

    def printTeams(self):
        print(f"Team A: {self.teamA}")
        print(f"Team B: {self.teamB}")

    def printState(self):
        for p in self.players:
            id = p.id

            if id in self.teamA:
                mod = "* "
            else:
                mod = ""

            print(f"{mod}{id}: \t{p.hand}")

    def getPlayer(self, id):
        for p in self.players:
            if p.id == id:
                return p

        return None

    def isOver(self):
        sets = [0] * 8

        for id in self.teamA:
            p = self.getPlayer(id)
            sets = [a + b for a, b in zip(sets, p.sets)]

        print("TEAM A SETS:")
        print(sets)
        print("\n")

        count = sets.count(6)

        if sets.count(0) + count == len(sets):
            if count > 4:
                self.winner = "Team A"
            elif count == 4:
                self.winner = "Tie"
            else:
                self.winner = "Team B"

            return True

        return False

    def playRound(self):
        if self.lastAsk is None:
            asker = secrets.choice(self.players)
        else:
            if self.lastAsk['success']:
                asker = self.lastAsk['from']
            else:
                asker = self.lastAsk['to']

        print(f"asker: {asker.id}, sets: {asker.sets}")
        ask = asker.pickAsk()
        to = self.players[self.ids.index(ask['to'])]

        print(f"{asker.id} asks {ask['to']} for the {card2str(ask['suite'], ask['value'])}")
        print(f"\texpected probability: {ask['prob']}")

        print("\tchecking ask")
        result = to.checkAsk(asker, ask['suite'], ask['value'])

        if result:
            print("\tsuccessful :)")
        else:
            print("\twomp womp :(")

        for p in self.players:
            if p is not asker and p is not to:
                print(p.id)
                p.callHeard(asker, to, ask['suite'], ask['value'], result)

        self.lastAsk = {"from": asker, "to": to, "suite": ask['suite'], "value": ask['value'], "success": result}

        print("\tasker parsing")
        asker.processResponse(self.lastAsk)

        print("\n")

if __name__ == '__main__':
    start = time.time()

    players = [1, 2, 3, 4, 5, 6]

    game = Game(players)

    game.printTeams()

    print("\n")

    game.printState()

    print("\n")

    for p in game.players:
        p.printProbs()

    print("\n")

    while not game.isOver():
        #input("continue?")

        try:
            game.playRound()
        except NoMovesException as e:
            print(e)
            print("\n")

            for p in game.players:
                p.printProbs()

            print("\n")
            break


        for p in game.players:
            p.printProbs()

        print("\n")

    game.printState()

    print("\n")

    print(f"Winner is {game.winner}")
    print(f"Completed in {time.time() - start} secs")
