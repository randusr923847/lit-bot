SET_SEG = {
    0: [1, 2, 3, 4, 5, 6],
    1: [0, 8, 9, 10, 11, 12],
    2: [14, 15, 16, 17, 18, 19],
    3: [13, 21, 22, 23, 24, 25],
    4: [27, 28, 29, 30, 31, 32],
    5: [26, 34, 35, 36, 37, 38],
    6: [40, 41, 42, 43, 44, 45],
    7: [39, 47, 48, 49, 50, 51]
}

def whatSet(suite, value):
    return 2 * suite + (0 if 2 <= value <= 7 else 1)

#def whatSet(suite, value):
#    return suite + (value // 8)

def ind2card(index):
    suite = index // 13
    value = (index % 13) + 1

    return suite, value

def card2ind(suite, value):
    return (suite * 13) + (value - 1)

def card2str(suite, value):
    s = ""

    if value == 1:
        s += "A"
    elif value == 11:
        s += "J"
    elif value == 12:
        s += "Q"
    elif value == 13:
        s += "K"
    else:
        s += str(value)

    if suite == 0:
        s += " of Spades"
    elif suite == 1:
        s += " of Clubs"
    elif suite == 2:
        s += " of Diamonds"
    elif suite == 3:
        s += " of Hearts"

    return s
