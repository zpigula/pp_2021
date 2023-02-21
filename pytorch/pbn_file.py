import sys
from collections import namedtuple

class Player:
    NORTH = 'N'
    SOUTH = 'S'
    EAST  = 'E'
    WEST  = 'W'

    POSITION = [NORTH,EAST,SOUTH,WEST]

    NS = [NORTH,SOUTH]
    EW = [EAST,WEST]

    NEXT = {NORTH:EAST,
            EAST:SOUTH,
            SOUTH:WEST,
            WEST:NORTH}

    TEAM = {NORTH:NS,
            SOUTH:NS,
            EAST:EW,
            WEST:EW}

class Card(namedtuple('Card','suit value')):
    """
    A Single Card, which has a suit and a value
    """
    __slots__ = ()
    """
    helps keep memory requirements low 
    by preventing the creation of instance dictionaries
    """

    def __str__(self):
        return self.suit + str(self.value)
    def __repr__(self):
        return self.__str__()

class Suit:
    CLUBS    = 'C'
    DIAMONDS = 'D'
    HEARTS   = 'H'
    SPADES   = 'S'
    SUITS = [CLUBS,DIAMONDS,HEARTS,SPADES]


class PBN(object):

    SUIT = 'SHDC'
    CARDRANK = dict(zip('23456789TJQKA',range(2,15)))
    POSITION = dict(zip('NESW',Player.POSITION))
    PLAYERINDEX = dict(zip('NESW',range(1,5)))
    
    def __init__(self, pbnFile):
        self.file_name = pbnFile

    def parsePBN(self,pbn):
        """Parses the given PBN string and extracts:
        
        * for each PBN tag, a dict of associated key/value pairs.
        * for each data section, a dict of key/data pairs.
            
        This method does not interpret the PBN string itself.
        
        @param pbn: a string containing PBN markup.
        @return: a tuple (tag values, section data, notes).
        """
        tagValues, sectionData, notes = {}, {}, {}
        tag = 'Default Tag'
        for line in pbn.splitlines():
            line.strip()  # Remove whitespace.

            if line.startswith('%'):  # A comment.
                pass  # Skip over comments.

            elif line.startswith('['):  # A tag.
                line = line.strip('[]')  # Remove leading [ and trailing ].
                # The key is the first word, the value is everything after.
                tag, value = line.split(' ', 1)
                tag = tag.capitalize()
                value = value.strip('\'\"')
                if tag == 'Note':
                    notes.setdefault(tag, [])
                    notes[tag].append(value)
                else:
                    tagValues[tag] = value

            else:  # Line follows tag, add line to data buffer for section.
                sectionData.setdefault(tag, '')
                sectionData[tag] += line + '\n'

        return tagValues, sectionData, notes

    def get_hands(self):

        my_file = open(self.file_name, 'r')

        #read text file into list
        pbn_data = my_file.read()
        tags, sections, notes = self.parsePBN(pbn_data)
        if 'Deal' in tags:
            first, cards = tags['Deal'].split(":")
            index = self.PLAYERINDEX[first.strip()] - 1

        order = Player.POSITION[index:] + Player.POSITION[:index]

        hands = {}
        for player, hand in zip(order, cards.strip().split()):
            hands[player] = {}
            for suit, suitcards in zip(self.SUIT, hand.split('.')):
                hands[player][suit] = ""
                for rank in suitcards:
                    card = Card(suit,self.CARDRANK[rank])
                    if hands[player][suit] != "":
                        hands[player][suit] += " "
                    if rank == "T":
                        rank ="10"
                    hands[player][suit] += rank
                if hands[player][suit] == "":
                    hands[player][suit] = "—"
        return hands

    def get_hands_in_rs_format(self):

        my_file = open(self.file_name, 'r')

        #read text file into list
        pbn_data = my_file.read()
        tags, sections, notes = self.parsePBN(pbn_data)
        if 'Deal' in tags:
            first, cards = tags['Deal'].split(":")
            index = self.PLAYERINDEX[first.strip()] - 1

        order = Player.POSITION[index:] + Player.POSITION[:index]

        hands = {}
        #print("\n*** hands ***")
        for player, hand in zip(order, cards.strip().split()):
            hands[player] = []
            for suit, suitcards in zip(self.SUIT, hand.split('.')):
                for rank in suitcards:
                    if rank == "T":
                        rank ="10"
                    hands[player].append(rank+suit)
            #print("player:",player,hands[player])

        return hands



