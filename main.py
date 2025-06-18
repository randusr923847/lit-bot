'''
Known problems:
    - very rarely infinite loops T-T

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
    def __init__(self, id, buds, opps, game, strat: 1):
        self.id = id
        self.hand = [0] * 52
        self.sets = [0] * 8
        self.buds = buds
        self.opps = opps
        self.out = []
        self.known_sets = []
        self.probabilities = []
        self.inds = {}
        self.strat = strat
        self.game = game
        self.lastAsk = None

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

        tots = [0] * 52

        for id, ind in self.inds.items():
            #print(f"id: {id}, p_ind: {ind}")

            if id in self.buds:
                mod = "* "
            else:
                mod = ""

            print(f"\t{mod}{id}: \t{[float(round(x, 1)) for x in self.probabilities[ind]]}")
            tots = list(np.array(self.probabilities[ind]) + np.array(tots))

        if tots.count(1) + tots.count(0) != len(tots):
            print(tots)
            input("oops???")

    def add(self, card):
        sui, val = card.SV()
        card_ind = card2ind(sui, val)
        self.hand[card_ind] = 1
        self.sets[whatSet(sui, val)] += 1

        for id, ind in self.inds.items():
            self.probabilities[ind][card_ind] = 0

    def giveToGame(self, card_ind):
        sui, val = ind2card(card_ind)
        set = whatSet(sui, val)
        self.hand[card_ind] = 0
        self.sets[set] -= 1
        self.game.sets[set] += 1
        self.game.hand[card_ind] = 1

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

    def adjustProbs(self, prev_prob, card_ind):
        #print(f"ADP: prev prob: {prev_prob}, card: {card_ind}")
        other_probs = []

        for id, ind in self.inds.items():
            #print(self.probabilities[ind][card_ind])
            if self.probabilities[ind][card_ind] != 0:
                other_probs.append(ind)

        #print(other_probs)
        #print("\n")

        additive = prev_prob / float(len(other_probs))

        for ind in other_probs:
            self.probabilities[ind][card_ind] += additive

    def askHeard(self, fromP, toP, sui, val, successful):
        from_ind = self.inds[fromP.id]
        to_ind = self.inds[toP.id]
        card_ind = card2ind(sui, val)
        card_set = whatSet(sui, val)

        self.known_sets[from_ind][card_set] = 1

        if successful:
            self.known_sets[to_ind][card_set] = 0

            for id, ind in self.inds.items():
                if id == fromP.id:
                    self.probabilities[ind][card_ind] = 1
                else:
                    self.probabilities[ind][card_ind] = 0
        elif self.hand[card_ind] == 0:
            prev_prob = self.probabilities[from_ind][card_ind] + self.probabilities[to_ind][card_ind]

            self.probabilities[from_ind][card_ind] = 0
            self.probabilities[to_ind][card_ind] = 0

            self.adjustProbs(prev_prob, card_ind)

    def playerOut(self, id):
        #print(f"player {self.id} is noting that {id} is out")
        p_ind = self.inds[id]

        if id in self.buds:
            self.buds.remove(id)
        else:
            self.opps.remove(id)

        self.out.append(id)

        for i in range(0, 52):
            prev_prob = self.probabilities[p_ind][i]

            if prev_prob != 0 and prev_prob != 1:
                self.probabilities[p_ind][i] = 0

                if self.hand[i] == 0 and self.game.hand[i] == 0:
                    #print(f"card {i}")
                    self.adjustProbs(prev_prob, i)

    def callHeard(self, move):
        set = move['call']
        inds = SET_SEG[set]

        for id, ind in self.inds.items():
            for card in inds:
                self.probabilities[ind][card] = 0

        #locs = move['locs']

        #for id, ind_list in locs.items():
        #    if id != self.id:
        #        p_ind = self.inds[id]

        #        for ind in ind_list:
        #            self.probabilities[p_ind][ind] = 0

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
        self.adjustProbs(prev_prob, card_ind)

        return False

    def processResponse(self, ask):
        card = card2ind(ask['suite'], ask['value'])
        card_set = whatSet(ask['suite'], ask['value'])
        n = len(self.probabilities)
        to_ind = self.inds[ask['to'].id]

        if ask['success']:
            self.known_sets[to_ind][card_set] = 0

            for p in range(0, n):
                self.probabilities[p][card] = 0
        else:
            prev_prob = self.probabilities[to_ind][card]

            self.probabilities[to_ind][card] = 0
            self.adjustProbs(prev_prob, card)

    def findBestNext(self):
        #print(f"{self.id} is finding next best")
        strengths = []

        if len(self.buds) == 0:
            if len(self.opps) == 0:
                return -1

            for id in self.opps:
                ind = self.inds[id]
                strengths.append(sum(self.probabilities[ind]))

            #print(strengths)
            #print(self.opps[np.argmin(strengths)])
            #print("\n")

            return self.opps[np.argmin(strengths)]

        for id in self.buds:
            ind = self.inds[id]
            strengths.append(sum(self.probabilities[ind]))

        #print(strengths)
        #print(self.buds[np.argmax(strengths)])
        #print("\n")

        return self.buds[np.argmax(strengths)]

    def isValidAsk(self, card_ind):
        sui, val = ind2card(card_ind)
        set = whatSet(sui, val)
        return self.sets[set] != 0 and val != 8 and self.hand[card_ind] == 0

    def pickRandom(self, options):
        options = options.copy()
        random.shuffle(options)

        to = secrets.choice(self.opps)
        to_ind = self.inds[to]
        card = options.pop()

        return card, to, -1

    def pickRandomNonZero(self, options):
        options = options.copy()
        random.shuffle(options)

        to = secrets.choice(self.opps)
        to_ind = self.inds[to]
        card = options.pop()

        while self.probabilities[to_ind][card] == 0 or not self.isValidAsk(card):
            if len(options) == 0:
                raise NoMovesException('Player has no good asks!')

            to = secrets.choice(self.opps)
            to_ind = self.inds[to]
            card = options.pop()

        return card, to, -1

    def callKnownSet(self, sets, buds_probs):
        call = secrets.choice(sets)
        locs = {}

        set_inds = SET_SEG[call]

        cards = [i for i, x in enumerate(self.hand) if x == 1 and i in set_inds]

        locs[self.id] = cards

        for ind, id in enumerate(self.buds):
            lst = [i for i, x in enumerate(buds_probs[ind]) if x == 1 and i in set_inds]

            if len(lst) != 0:
                locs[id] = lst

        if len(cards) == self.hand.count(1):
            return {'type': 1, 'call': call, 'locs': locs}

        return {'type': 1, 'call': call, 'locs': locs}

    def pickOrCall(self, possible, possible_sets):
        sets = []
        buds_probs = []

        for id in self.buds:
            buds_probs.append(self.probabilities[self.inds[id]])

        for i, set in enumerate(self.sets):
            if set > 0:
                count = 0
                set_inds = SET_SEG[i]

                for probs in buds_probs:
                    count += np.array(probs)[set_inds].tolist().count(1.0)

                if count + set == 6:
                    sets.append(i)

        if len(sets) > 0 and secrets.choice([True]):
            self.lastAsk = None
            return self.callKnownSet(sets, buds_probs)

        return self.pickAsk(possible, possible_sets, sets)

    def callOwnSet(self):
        locs = {}
        cards = []
        call = -1

        for i, n in enumerate(self.sets):
            if n == 6:
                call = i
                cards.extend(SET_SEG[i])

                break

        locs[self.id] = cards

        if len(cards) == self.hand.count(1):
            return {'type': 1, 'call': call, 'locs': locs}

        return {'type': 1, 'call': call, 'locs': locs}

    def callWithinTeam(self):
        sets = []
        sureness = []
        buds_probs = []

        for id in self.buds:
            buds_probs.append(self.probabilities[self.inds[id]])

        for i, set in enumerate(self.sets):
            if set > 0:
                count = 0
                set_inds = SET_SEG[i]

                for probs in buds_probs:
                    count += np.array(probs)[set_inds].tolist().count(1.0)

                sets.append(i)
                sureness.append(count + set)

        call = sets[np.argmax(sureness)]

        locs = {}
        set_inds = SET_SEG[call]

        locs[self.id] = []

        for ind in set_inds:
            if self.hand[ind] == 1:
                locs[self.id].append(ind)
            else:
                opts = []

                for id in self.buds:
                    opts.append(self.probabilities[self.inds[id]][ind])

                bud_id = self.buds[np.argmax(opts)]

                if bud_id in locs:
                    locs[bud_id].append(ind)
                else:
                    locs[bud_id] = [ind]


        if len(locs[self.id]) == self.hand.count(1):
            return {'type': 1, 'call': call, 'locs': locs}

        return {'type': 1, 'call': call, 'locs': locs}

    def pickInfoAsk(self, buds_probs, possible):
        min_knowledges = []

        print(f"possible: {possible}")

        for ind in possible:
            card_probs = []

            for probs in buds_probs:
                card_probs.append(probs[ind])

            min_knowledges.append(max(card_probs))

        print(f"minK: {min_knowledges}")

        min_know = min_knowledges[np.argmin(min_knowledges)]

        choices = [possible[i] for i, k in enumerate(min_knowledges) if k == min_know]

        card = secrets.choice(choices)
        to = secrets.choice(self.opps)

        return card, to, -2

    def pickAsk(self, possible, possible_sets, sets):
        if self.strat == 1:
            options_ids = []
            options_cards = []
            options_prob = []

            for id in self.opps:
                p_ind = self.inds[id]
                #print(f"\nid: {id}, p_ind: {p_ind}")
                #print(possible)

                valid_prob_cards = []
                valid_probs = []

                for i, c_ind in enumerate(possible):
                    c_prob = self.probabilities[p_ind][c_ind]

                    if c_prob != 0:
                        additive = self.known_sets[p_ind][possible_sets[i]] * 1.2
                        valid_probs.append(c_prob + additive)
                        valid_prob_cards.append(c_ind)

                #print(valid_probs)
                #print(valid_prob_cards)

                #print("\n")

                if len(valid_probs) == 0:
                    continue

                best_ind = np.argmax(valid_probs)
                card = valid_prob_cards[best_ind]
                prob = valid_probs[best_ind]
                #print(f"choice: {card}, {valid_probs[best_ind]}")

                #prob = self.probabilities[p_ind][card]

                #if prob != 0:
                #    sui, val = ind2card(card)
                #    additive = self.known_sets[p_ind][whatSet(sui, val)] * 1.2
                #else:
                #    additive = 0

                #print("Ad.", additive)

                #prob = prob + additive

                #print("Pb.", prob)

                #print("\n")

                options_ids.append(id)
                options_cards.append(card)
                options_prob.append(prob)

            options = list(zip(options_ids, options_cards, options_prob))
            options = sorted(options, key=lambda x: x[2])

            if len(options) == 0:
                buds_probs = []

                for id in self.buds:
                    buds_probs.append(self.probabilities[self.inds[id]])

                for i, set in enumerate(self.sets):
                    if set > 0:
                        count = 0
                        set_inds = SET_SEG[i]
                        unknown_cards = []

                        for ind in set_inds:
                            probs = [probs[ind] for probs in buds_probs]

                            if self.hand[ind] == 1 or probs.count(1) == 1:
                                count += 1
                            else:
                                unknown_cards.append(ind)

                        if count == 5:
                            for c in unknown_cards:
                                known = []
                                for id in self.buds:
                                    idx = self.inds[id]
                                    if self.known_sets[idx][i] == 1:
                                        known.append(idx)

                                if len(known) == 1:
                                    for id in self.buds:
                                        self.probabilities[idx][c] = 0.0

                                    self.probabilities[known[0]][c] = 1.0

                            return self.callKnownSet([i], buds_probs)

                try:
                    card, to, prob = self.pickRandomNonZero(possible)
                except NoMovesException:
                    card, to, prob = self.pickInfoAsk(buds_probs, possible)
            else:
                card = options[-1][1]

                #while len(options) > 1 and not self.isValidAsk(card):
                #    del options[-1]
                #    card = options[-1][1]

                #if not self.isValidAsk(card):
                #    card, to, prob = self.pickRandom(possible)
                #else:
                to = options[-1][0]
                prob = options[-1][2]

            suite, value = ind2card(card)

            #if self.lastAsk and self.lastAsk == {'type': 0, 'to': to, 'suite': suite, 'value': value, 'prob': prob}:
            #    print(f"{self.id} asks {to} for the {card2str(suite, value)}")
            #    print(f"\texpected probability: {prob}")
            #    input("continue?")

            #self.lastAsk = {'type': 0, 'to': to, 'suite': suite, 'value': value, 'prob': prob}
        else:
            card, to, prob = self.pickRandom(possible)

        suite, value = ind2card(card)

        return {'type': 0, 'to': to, 'suite': suite, 'value': value, 'prob': prob}

    def ask(self):
        possible = []
        possible_sets = []

        for i, set in enumerate(self.sets):
            if set > 0:
                valid_set_list = [ind for ind in SET_SEG[i] if self.hand[ind] == 0]
                valid_set_sets = [i for ind in SET_SEG[i] if self.hand[ind] == 0]
                possible.extend(valid_set_list)
                possible_sets.extend(valid_set_sets)

        print(f"\nPossible: {possible}\n")

        if len(possible) == 0:
            self.lastAsk = None
            return self.callOwnSet()
        elif len(self.opps) == 0:
            self.lastAsk = None
            return self.callWithinTeam()
        else:
            return self.pickOrCall(possible, possible_sets)

class Game:
    def __init__(self, players, use_strat=0):
        self.ids = list(set(players))
        self.teamA = self.ids[::2]
        self.teamAmoves = 0
        self.teamAcorrect = 0
        self.teamB = self.ids[1::2]
        self.teamBmoves = 0
        self.teamBcorrect = 0
        self.players = []
        self.lastAsk = None
        self.winner = None
        self.teamAsets = [0] * 8
        self.teamBsets = [0] * 8
        self.outPlayers = []
        self.sets = [0] * 8
        self.hand = [0] * 52

        for id in self.ids:
            if id in self.teamA:
                buds = self.teamA.copy()
                buds.remove(id)
                opps = self.teamB.copy()
                strat = use_strat                   # change to have one team use random strat
            else:
                buds = self.teamB.copy()
                buds.remove(id)
                opps = self.teamA.copy()
                strat = 1

            self.players.append(Player(id, buds, opps, self, strat))

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

    def numCards(self, id):
        return self.getPlayer(id).hand.count(1)

    def isOver(self):
        print("TEAM A SETS:")
        print(self.teamAsets)

        print("TEAM B SETS:")
        print(self.teamBsets)
        print("\n")

        over = self.teamAsets.count(1) + self.teamBsets.count(1) == 8

        if over:
            a_sets = self.teamAsets.count(1)
            b_sets = self.teamBsets.count(1)

            if a_sets > b_sets:
                self.winner = "Team A"
            elif a_sets == b_sets:
                self.winner = "Tie"
            else:
                self.winner = "Team B"

        return over

    def checkCall(self, set, locs):
        set_list = SET_SEG[set].copy()
        print(set_list)
        print(locs)

        for id, inds in locs.items():
            p = self.getPlayer(id)

            for ind in inds:
                 if ind in set_list and p.hand[ind] == 1:
                     set_list.remove(ind)
                 else:
                     return False

        if len(set_list) != 0:
            return False

        for id, inds in locs.items():
            p = self.getPlayer(id)

            for ind in inds:
                p.giveToGame(ind)

        return True

    def findSetLocs(self, set_ind):
        set_list = SET_SEG[set_ind].copy()
        locs = {}

        for p in self.players:
            if p.sets[set_ind] != 0:
                p_list = []

                rmed = []

                for ind in set_list:
                    if p.hand[ind] == 1:
                        p_list.append(ind)
                        rmed.append(ind)

                set_list = list(set(set_list) - set(rmed))
                locs[p.id] = p_list

        return locs

    def checkOuts(self):
        to_remove = []

        for p in self.players:
            if 1 not in p.hand:
                print(f"{p.id} is out of cards and the game!")
                self.outPlayers.append(p)
                to_remove.append(p)

                if p.id in self.teamA:
                    self.teamA.remove(p.id)
                else:
                    self.teamB.remove(p.id)

        for p in to_remove:
            self.players.remove(p)

        for p in self.players:
            for out in to_remove:
                p.playerOut(out.id)

        return to_remove

    def playRound(self):
        if self.lastAsk is None:
            asker = secrets.choice(self.players)
        else:
            if self.lastAsk['success']:
                asker = self.lastAsk['from']
            else:
                asker = self.lastAsk['to']

        if asker.id in self.teamA:
            self.teamAmoves += 1
        else:
            self.teamBmoves += 1

        print(f"game sets: {self.sets}")
        print(f"asker: {asker.id}, sets: {asker.sets}")
        move = asker.ask()

        if move['type'] == 1:
            print(f"{asker.id} calls set {move['call']}")

            print("\tchecking call")
            result = self.checkCall(move['call'], move['locs'])
            print(result)

            move['from'] = asker

            if not result:
                print("\twomp womp :(")
                move['success'] = False

                move['locs'] = self.findSetLocs(move['call'])
                self.checkCall(move['call'], move['locs'])

                for p in self.players:
                    p.callHeard(move)

                self.checkOuts()

                to = None

                if asker.id in self.teamA:
                    if len(self.teamB) > 0:
                        to = self.getPlayer(secrets.choice(self.teamB))
                    elif len(self.players) > 0:
                        to = secrets.choice(self.players)

                    self.teamBsets[move['call']] = 1
                else:
                    if len(self.teamA) > 0:
                        to = self.getPlayer(secrets.choice(self.teamA))
                    elif len(self.players) > 0:
                        to = secrets.choice(self.players)

                    self.teamAsets[move['call']] = 1


                self.lastAsk = {'success': False, 'to': to}
            else:
                print("\tsuccessful! :)")
                if asker.id in self.teamA:
                    self.teamAsets[move['call']] = 1
                    self.teamAcorrect += 1
                else:
                    self.teamBsets[move['call']] = 1
                    self.teamBcorrect += 1

                out = asker.hand.count(1) == 0
                move['success'] = True

                for p in self.players:
                    p.callHeard(move)

                to_remove = self.checkOuts()

                if out:
                    to_remove.remove(asker)

                    for p in to_remove:
                        asker.playerOut(p.id)

                    self.lastAsk = {"success": True, "from": self.getPlayer(asker.findBestNext())}
                else:
                    self.lastAsk = {"success": True, "from": asker}
        elif move['type'] == 0:
            ask = move
            to = self.getPlayer(ask['to'])

            print(f"{asker.id} asks {ask['to']} for the {card2str(ask['suite'], ask['value'])}")
            print(f"\texpected probability: {ask['prob']}")

            print("\tchecking ask")
            result = to.checkAsk(asker, ask['suite'], ask['value'])

            if result:
                print("\tsuccessful :)")

                self.checkOuts()

                if asker.id in self.teamA:
                    self.teamAcorrect += 1
                else:
                    self.teamBcorrect += 1
            else:
                print("\twomp womp :(")

            for p in self.players:
                if p is not asker and p is not to:
                    #print(p.id)
                    p.askHeard(asker, to, ask['suite'], ask['value'], result)

            self.lastAsk = {"from": asker, "to": to, "suite": ask['suite'], "value": ask['value'], "success": result}

            print("\tasker parsing")
            asker.processResponse(self.lastAsk)

            print("\n")

def runGame(do_print=False, strat=0):
    start = time.time()
    count = 0

    players = [1, 2, 3, 4, 5, 6]

    game = Game(players, strat)

    if do_print:
        game.printTeams()

        print("\n")

    #game.printState()

    #print("\n")

    #for p in game.players:
    #    p.printProbs()

    #print("\n")

    while not game.isOver():
        #input("continue?")

        #print(f"COUNT: {count}")

        count += 1
        game.playRound()

        #for p in game.players:
        #    p.printProbs()

        #print("\n")

        #game.printState()

        #print("\n")

    #game.printState()

    ret = {}

    if game.winner == "Team A":
        ret['outcome'] = -1
    elif game.winner == "Tie":
        ret['outcome'] = 0
    else:
        ret['outcome'] = 1

    ret['time'] = time.time() - start
    ret['accA'] = game.teamAcorrect / game.teamAmoves
    ret['accB'] = game.teamBcorrect / game.teamBmoves
    ret['setsA'] = game.teamAsets.count(1)
    ret['setsB'] = game.teamBsets.count(1)
    ret['moves'] = count

    if do_print:
        print("\n")

        print(f"Winner is {game.winner}")
        print(f"Completed in {ret['time']} secs with {count} moves")
        print(f"Team A accuracy: {ret['accA']}")
        print(f"Team B accuracy: {ret['accB']}")

    return ret


def avg(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    start = time.time()
    n_runs = 100
    strat = 1
    strat_name = "Greedy 0" if strat else "Random"

    outcomes = []
    times = []
    accAs = []
    accBs = []
    setsA = []
    setsB = []
    moves = []

    for i in range(n_runs):
        info = runGame(strat=strat)
        outcomes.append(info['outcome'])
        times.append(info['time'])
        accAs.append(info['accA'])
        accBs.append(info['accB'])
        setsA.append(info['setsA'])
        setsB.append(info['setsB'])
        moves.append(info['moves'])
        print(f"RUN {i}")

    print(f"Ran {n_runs} times:")
    print(f"\tAvg Win rate: {outcomes.count(1) / len(outcomes)}\tAvg Tie rate: {outcomes.count(0) / len(outcomes)}\tAvg Loss rate: {outcomes.count(-1) / len(outcomes)}")
    print(f"\t{strat_name} Avg Sets: {avg(setsA)}\tGreedy Avg Sets: {avg(setsB)}")
    print(f"\t{strat_name} Avg Accuracy: {avg(accAs)}")
    print(f"\tGreedy Avg Accuracy: {avg(accBs)}")
    print(f"\tAvg # Moves: {avg(moves)}")
    print(f"\tAvg Time: {avg(times)} secs")
    print(f"Total Time: {time.time() - start} secs")
