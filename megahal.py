# MegaHAL for Python 3.
#
# Originally written by Chris Jones in 2010 as py-megahal and released under
# the BSD license, forked by Kyle Maxwell in 2015 and then by Pavel Mises in
# 2026, who ported it to Python 3. The changes made by Pavel Mises are released
# under the WTFPL, see the LICENSE file.
#
# Copyright (c) 2010, Chris Jones <cjones@gruntle.org>
# Copyright (c) 2015, Kyle Maxwell <krmaxwell@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright © 2026 Pavel Mises <id@dqd.cz>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.

"""Python implementation of the MegaHAL Markov chain chatbot."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import shelve
import sys
from time import monotonic
from typing import Iterable, Sequence, Union

__version__ = "1.0.0"
__author__ = "dqd <id@dqd.cz>"
__license__ = "WTFPL AND BSD-2-Clause"
__all__ = [
    "DEFAULT_BRAINFILE",
    "DEFAULT_ORDER",
    "DEFAULT_TIMEOUT",
    "Dictionary",
    "MegaHAL",
    "Tree",
    "__version__",
]

DEFAULT_ORDER = 5
DEFAULT_BRAINFILE = os.path.expanduser("~/.pymegahal-brain")
DEFAULT_TIMEOUT = 1.0

API_VERSION = "1.0"
END_WORD = "<FIN>"
ERROR_WORD = "<ERROR>"
UNKNOWN_REPLY = "I don't know enough to answer you yet!"

PathLike = Union[str, "os.PathLike[str]"]

# fmt: off
DEFAULT_BANWORDS = ['A', 'ABILITY', 'ABLE', 'ABOUT', 'ABSOLUTE', 'ABSOLUTELY', 'ACROSS', 'ACTUAL', 'ACTUALLY', 'AFTER',
                    'AGAIN', 'AGAINST', 'AGO', 'AGREE', 'ALL', 'ALMOST', 'ALONG', 'ALREADY', 'ALTHOUGH', 'ALWAYS',
                    'AN', 'AND', 'ANOTHER', 'ANY', 'ANYHOW', 'ANYTHING', 'ANYWAY', 'ARE', "AREN'T", 'AROUND', 'AS',
                    'AWAY', 'BACK', 'BAD', 'BE', 'BEEN', 'BEFORE', 'BEHIND', 'BEING', 'BELIEVE', 'BELONG', 'BEST',
                    'BETWEEN', 'BIG', 'BIGGER', 'BIGGEST', 'BIT', 'BOTH', 'BUDDY', 'BUT', 'BY', 'CALL', 'CALLED',
                    'CAME', 'CAN', "CAN'T", 'CANNOT', 'CARE', 'CARING', 'CASE', 'CATCH', 'CAUGHT', 'CERTAIN',
                    'CHANGE', 'CLOSE', 'CLOSER', 'COME', 'COMING', 'COMMON', 'CONSTANT', 'CONSTANTLY', 'COULD',
                    'DAY', 'DAYS', 'DERIVED', 'DESCRIBE', 'DESCRIBES', 'DETERMINE', 'DETERMINES', 'DID', "DIDN'T",
                    'DOES', "DOESN'T", 'DOING', "DON'T", 'DONE', 'DOUBT', 'DOWN', 'EACH', 'EARLIER', 'EARLY', 'ELSE',
                    'ESPECIALLY', 'EVEN', 'EVER', 'EVERY', 'EVERYBODY', 'EVERYONE', 'EVERYTHING', 'FACT', 'FAIR',
                    'FAR', 'FELLOW', 'FEW', 'FIND', 'FINE', 'FOR', 'FORM', 'FOUND', 'FROM', 'FULL', 'FURTHER', 'GAVE',
                    'GETTING', 'GIVE', 'GIVEN', 'GIVING', 'GO', 'GOING', 'GONE', 'GOOD', 'GOT', 'GOTTEN', 'GREAT',
                    'HAS', "HASN'T", 'HAVE', "HAVEN'T", 'HAVING', 'HELD', 'HERE', 'HIGH', 'HOLD', 'HOLDING', 'HOW',
                    'IN', 'INDEED', 'INSIDE', 'INSTEAD', 'INTO', 'IS', "ISN'T", 'IT', "IT'S", 'ITS', 'JUST', 'KEEP',
                    'KNEW', 'KNOW', 'KNOWN', 'LARGE', 'LARGER', 'LARGETS', 'LAST', 'LATE', 'LATER', 'LEAST', 'LESS',
                    "LET'S", 'LEVEL', 'LIKES', 'LITTLE', 'LONG', 'LONGER', 'LOOK', 'LOOKED', 'LOOKING', 'LOOKS', 'LOW',
                    'MAKE', 'MAKING', 'MANY', 'MATE', 'MAY', 'MAYBE', 'MEAN', 'MEET', 'MENTION', 'MERE', 'MIGHT',
                    'MORE', 'MORNING', 'MOST', 'MOVE', 'MUCH', 'MUST', 'NEAR', 'NEARER', 'NEVER', 'NEXT', 'NICE',
                    'NONE', 'NOON', 'NOONE', 'NOT', 'NOTE', 'NOTHING', 'NOW', 'OBVIOUS', 'OF', 'OFF', 'ON', 'ONCE',
                    'ONTO', 'OPINION', 'OR', 'OTHER', 'OUR', 'OUT', 'OVER', 'OWN', 'PART', 'PARTICULAR',
                    'PERHAPS', 'PERSON', 'PIECE', 'PLACE', 'PLEASANT', 'PLEASE', 'POPULAR', 'PREFER', 'PRETTY', 'PUT',
                    'REAL', 'REALLY', 'RECEIVE', 'RECEIVED', 'RECENT', 'RECENTLY', 'RELATED', 'RESULT', 'RESULTING',
                    'SAID', 'SAME', 'SAW', 'SAY', 'SAYING', 'SEE', 'SEEM', 'SEEMED', 'SEEMS', 'SEEN', 'SELDOM',
                    'SET', 'SEVERAL', 'SHALL', 'SHORT', 'SHORTER', 'SHOULD', 'SHOW', 'SHOWS', 'SIMPLE', 'SIMPLY',
                    'SO', 'SOME', 'SOMEONE', 'SOMETHING', 'SOMETIME', 'SOMETIMES', 'SOMEWHERE', 'SORT', 'SORTS',
                    'SPENT', 'STILL', 'STUFF', 'SUCH', 'SUGGEST', 'SUGGESTION', 'SUPPOSE', 'SURE', 'SURELY',
                    'SURROUNDS', 'TAKE', 'TAKEN', 'TAKING', 'TELL', 'THAN', 'THANK', 'THANKS', 'THAT', "THAT'S",
                    'THE', 'THEIR', 'THEM', 'THEN', 'THERE', 'THEREFORE', 'THESE', 'THEY', 'THING', 'THINGS', 'THIS',
                    'THOUGH', 'THOUGHTS', 'THOUROUGHLY', 'THROUGH', 'TINY', 'TO', 'TODAY', 'TOGETHER', 'TOLD',
                    'TOO', 'TOTAL', 'TOTALLY', 'TOUCH', 'TRY', 'TWICE', 'UNDER', 'UNDERSTAND', 'UNDERSTOOD', 'UNTIL',
                    'US', 'USED', 'USING', 'USUALLY', 'VARIOUS', 'VERY', 'WANT', 'WANTED', 'WANTS', 'WAS', 'WATCH',
                    'WAYS', 'WE', "WE'RE", 'WELL', 'WENT', 'WERE', 'WHAT', "WHAT'S", 'WHATEVER', 'WHATS', 'WHEN',
                    "WHERE'S", 'WHICH', 'WHILE', 'WHILST', 'WHO', "WHO'S", 'WHOM', 'WILL', 'WISH', 'WITH', 'WITHIN',
                    'WONDERFUL', 'WORSE', 'WORST', 'WOULD', 'WRONG', 'YESTERDAY', 'YET']

DEFAULT_AUXWORDS = ['DISLIKE', 'HE', 'HER', 'HERS', 'HIM', 'HIS', 'I', "I'D", "I'LL", "I'M", "I'VE", 'LIKE', 'ME',
                    'MY', 'MYSELF', 'ONE', 'SHE', 'THREE', 'TWO', 'YOU', "YOU'D", "YOU'LL", "YOU'RE", "YOU'VE", 'YOUR',
                    'YOURSELF']

DEFAULT_SWAPWORDS = {"YOU'RE": "I'M", "YOU'D": "I'D", 'HATE': 'LOVE', 'YOUR': 'MY', "I'LL": "YOU'LL", 'NO': 'YES',
                     'WHY': 'BECAUSE', 'YOU': 'ME', 'LOVE': 'HATE', 'I': 'YOU', 'MINE': 'YOURS', 'YOURSELF': 'MYSELF',
                     'DISLIKE': 'LIKE', "I'M": "YOU'RE", 'ME': 'YOU', 'MYSELF': 'YOURSELF', 'LIKE': 'DISLIKE',
                     "I'D": "YOU'D", "YOU'VE": "I'VE", 'YES': 'NO', 'MY': 'YOUR'}
# fmt: on


class Tree:
    """Node of the Markov model. Each child is a symbol that followed this node's symbol."""

    def __init__(self, symbol: int = 0) -> None:
        self.symbol = symbol
        self.usage = 0
        self.count = 0
        self.children: list[Tree] = []

    def add_symbol(self, symbol: int) -> Tree:
        node = self.get_child(symbol)
        node.count += 1
        self.usage += 1
        return node

    def get_child(self, symbol: int, add: bool = True) -> Tree | None:
        for child in self.children:
            if child.symbol == symbol:
                return child
        if add:
            child = Tree(symbol)
            self.children.append(child)
            return child
        return None


class Dictionary(list):
    """List of words. The position of a word in the list is its symbol."""

    def add_word(self, word: str) -> int:
        try:
            return self.index(word)
        except ValueError:
            self.append(word)
            return len(self) - 1

    def find_word(self, word: str) -> int:
        try:
            return self.index(word)
        except ValueError:
            return 0


class Context:
    """Window of the most recently visited nodes, one per order of the Markov chain.

    With ``learn=True`` walking the context extends the tree (the model is trained),
    otherwise it just follows existing branches (the model is only read).
    """

    def __init__(self, brain: Brain, tree: Tree, learn: bool = False) -> None:
        self.brain = brain
        self.learn = learn
        self.used_key = False
        self.nodes: list[Tree | None] = [None] * (brain.order + 2)
        self.nodes[0] = tree

    @property
    def root(self) -> Tree:
        return self.nodes[0]  # type: ignore[return-value]

    def update(self, symbol: int) -> None:
        for i in range(self.brain.order + 1, 0, -1):
            node = self.nodes[i - 1]
            if node is not None:
                self.nodes[i] = node.add_symbol(symbol) if self.learn else node.get_child(symbol, add=False)

    def seed(self, keys: Sequence[str]) -> int:
        """Pick the first symbol of a reply, preferably one of the keywords."""
        if keys:
            i = random.randrange(len(keys))
            for key in list(keys[i:]) + list(keys[:i]):
                if key not in self.brain.auxwords:
                    try:
                        return self.brain.dictionary.index(key)
                    except ValueError:
                        pass
        if self.root.children:
            return random.choice(self.root.children).symbol
        return 0

    def babble(self, keys: Sequence[str], replies: Sequence[str]) -> int:
        """Pick the next symbol of a reply by a weighted random walk of the deepest known context."""
        node = self.root
        for candidate in self.nodes[: self.brain.order + 1]:
            if candidate is not None:
                node = candidate
        if not node.children:
            return 0
        i = random.randrange(len(node.children))
        count = random.randrange(node.usage)
        symbol = 0
        while count >= 0:
            child = node.children[i]
            symbol = child.symbol
            word = self.brain.dictionary[symbol]
            if word in keys and (self.used_key or word not in self.brain.auxwords) and word not in replies:
                self.used_key = True
                break
            count -= child.count
            i = 0 if i >= len(node.children) - 1 else i + 1
        return symbol


class Brain:
    """The model and the persistent storage behind it."""

    def __init__(self, order: int, file: PathLike, timeout: float) -> None:
        self.timeout = timeout
        self.closed = False
        self.db = shelve.open(os.fspath(file), protocol=4, writeback=True)  # noqa: SIM115
        api = self.db.setdefault("api", API_VERSION)
        if api != API_VERSION:
            raise ValueError(f"This brain has an incompatible API version: {api} != {API_VERSION}")
        if self.db.setdefault("order", order) != order:
            raise ValueError(f"This brain already has an order of {self.db['order']}")
        self.forward: Tree = self.db.setdefault("forward", Tree())
        self.backward: Tree = self.db.setdefault("backward", Tree())
        self.dictionary: Dictionary = self.db.setdefault("dictionary", Dictionary())
        self.error_symbol = self.dictionary.add_word(ERROR_WORD)
        self.end_symbol = self.dictionary.add_word(END_WORD)
        self.banwords: Dictionary = self.db.setdefault("banwords", Dictionary(DEFAULT_BANWORDS))
        self.auxwords: Dictionary = self.db.setdefault("auxwords", Dictionary(DEFAULT_AUXWORDS))
        self.swapwords: dict[str, str] = self.db.setdefault("swapwords", dict(DEFAULT_SWAPWORDS))

    @property
    def order(self) -> int:
        return self.db["order"]

    @staticmethod
    def get_words_from_phrase(phrase: str) -> list[str]:
        """Split a phrase into words and separators, both are kept. The last token is always a sentence end."""
        phrase = phrase.upper()
        if not phrase:
            return []

        def boundary(string: str, position: int) -> bool:
            if position == len(string):
                return True
            current, previous = string[position], string[position - 1]
            if current == "'" and previous.isalpha() and position + 1 < len(string) and string[position + 1].isalpha():
                return False
            if position > 1 and previous == "'" and string[position - 2].isalpha() and current.isalpha():
                return False
            if current.isalpha() != previous.isalpha():
                return True
            return current.isdigit() != previous.isdigit()

        words = []
        while phrase:
            offset = 1
            while not boundary(phrase, offset):
                offset += 1
            words.append(phrase[:offset])
            phrase = phrase[offset:]

        if words[-1][0].isalnum():
            words.append(".")
        elif words[-1][-1] not in "!.?":
            words[-1] = "."
        return words

    def communicate(self, phrase: str, learn: bool = True, reply: bool = True) -> str | None:
        words = self.get_words_from_phrase(phrase)
        if learn:
            self.learn(words)
        if reply:
            return self.get_reply(words)
        return None

    def learn(self, words: Sequence[str]) -> None:
        if len(words) <= self.order:
            return
        context = Context(self, self.forward, learn=True)
        for word in words:
            context.update(self.dictionary.add_word(word))
        context.update(self.end_symbol)
        context = Context(self, self.backward, learn=True)
        for word in reversed(words):
            context.update(self.dictionary.index(word))
        context.update(self.end_symbol)

    def get_reply(self, words: Sequence[str]) -> str:
        keywords = self.make_keywords(words)
        output = self.generate_replywords()
        if not output or output == list(words):
            output = self.get_words_from_phrase(UNKNOWN_REPLY)

        max_surprise = -1.0
        basetime = monotonic()
        while monotonic() - basetime < self.timeout:
            reply = self.generate_replywords(keywords)
            surprise = self.evaluate_reply(keywords, reply)
            if reply and surprise > max_surprise and reply != list(words):
                max_surprise = surprise
                output = reply

        return "".join(output).capitalize()

    def evaluate_reply(self, keys: Sequence[str], words: Sequence[str]) -> float:
        """Measure how surprising a reply is, given the keywords. Higher is better."""
        if not words:
            return 0.0
        num = 0
        entropy = 0.0
        for tree, sequence in ((self.forward, words), (self.backward, reversed(words))):
            context = Context(self, tree)
            for word in sequence:
                symbol = self.dictionary.index(word)
                if word in keys:
                    probability = 0.0
                    count = 0
                    num += 1
                    for node in context.nodes[: self.order]:
                        if node is not None:
                            child = node.get_child(symbol, add=False)
                            if child is not None:
                                probability += child.count / node.usage
                            count += 1
                    if count and probability > 0:
                        entropy -= math.log(probability / count)
                context.update(symbol)

        if num >= 8:
            entropy /= math.sqrt(num - 1)
        if num >= 16:
            entropy /= num
        return entropy

    def generate_replywords(self, keys: Sequence[str] = ()) -> list[str]:
        """Generate a reply as a list of words, growing it forwards from a seed and then backwards."""
        replies: list[str] = []
        context = Context(self, self.forward)
        symbol = context.seed(keys)
        while symbol not in (self.error_symbol, self.end_symbol):
            replies.append(self.dictionary[symbol])
            context.update(symbol)
            symbol = context.babble(keys, replies)

        used_key = context.used_key
        context = Context(self, self.backward)
        context.used_key = used_key
        for i in range(min(len(replies) - 1, self.order), -1, -1):
            context.update(self.dictionary.index(replies[i]))
        while True:
            symbol = context.babble(keys, replies)
            if symbol in (self.error_symbol, self.end_symbol):
                break
            replies.insert(0, self.dictionary[symbol])
            context.update(symbol)

        return replies

    def make_keywords(self, words: Iterable[str]) -> Dictionary:
        """Pick the words worth replying to: known, not banned, with auxiliary words only as a supplement."""
        words = [self.swapwords.get(word, word) for word in words]
        keys = Dictionary()
        for word in words:
            if (
                self.dictionary.find_word(word) != self.error_symbol
                and word[0].isalnum()
                and word not in self.banwords
                and word not in self.auxwords
                and word not in keys
            ):
                keys.append(word)

        if keys:
            for word in words:
                if (
                    self.dictionary.find_word(word) != self.error_symbol
                    and word[0].isalnum()
                    and word in self.auxwords
                    and word not in keys
                ):
                    keys.append(word)

        return keys

    def sync(self) -> None:
        self.db.sync()

    def close(self) -> None:
        if not self.closed:
            self.db.close()
            self.closed = True

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class MegaHAL:
    """Conversation simulator which learns from what it is told and replies with Markov chain babble.

    The brain is stored in a :mod:`shelve` database at ``brainfile``. ``order`` is the order of the
    Markov chain and ``timeout`` is the number of seconds spent looking for the best reply.
    Can be used as a context manager, the brain is closed on exit.
    """

    def __init__(
        self,
        order: int | None = None,
        brainfile: PathLike | None = None,
        timeout: float | None = None,
    ) -> None:
        if order is None:
            order = DEFAULT_ORDER
        if brainfile is None:
            brainfile = DEFAULT_BRAINFILE
        if timeout is None:
            timeout = DEFAULT_TIMEOUT
        self.__brain = Brain(order, brainfile, timeout)

    def __enter__(self) -> MegaHAL:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def banwords(self) -> Dictionary:
        """List of words which cannot be used as keywords."""
        return self.__brain.banwords

    @property
    def auxwords(self) -> Dictionary:
        """List of words which can be used as keywords only in order to supplement other keywords."""
        return self.__brain.auxwords

    @property
    def swapwords(self) -> dict[str, str]:
        """The word on the left is changed to the word on the right when used as a keyword."""
        return self.__brain.swapwords

    def train(self, file: PathLike, encoding: str = "utf-8") -> None:
        """Train the brain with a text file, each line is a phrase. Lines starting with ``#`` are skipped."""
        with open(file, encoding=encoding, errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.learn(line)

    def learn(self, phrase: str) -> None:
        """Learn from a phrase."""
        self.__brain.communicate(phrase, reply=False)

    def get_reply(self, phrase: str) -> str:
        """Learn from a phrase and get a reply to it."""
        return self.__brain.communicate(phrase)  # type: ignore[return-value]

    def get_reply_nolearn(self, phrase: str) -> str:
        """Get a reply to a phrase without learning from it."""
        return self.__brain.communicate(phrase, learn=False)  # type: ignore[return-value]

    def interact(self) -> None:
        """Have a friendly chat session. Ctrl-D (or Ctrl-C) to exit."""
        while True:
            try:
                phrase = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if phrase:
                print(self.get_reply(phrase))

    def sync(self) -> None:
        """Flush any changes to disk."""
        self.__brain.sync()

    def close(self) -> None:
        """Flush any changes to disk and close the brain."""
        self.__brain.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Command line interface: optionally train the brain, then chat."""
    parser = argparse.ArgumentParser(prog="megahal", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-b",
        "--brain",
        dest="brainfile",
        metavar="<file>",
        default=DEFAULT_BRAINFILE,
        help="location of the brain (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--order",
        metavar="<int>",
        type=int,
        default=DEFAULT_ORDER,
        help="order of the Markov chain (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        metavar="<float>",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="how long to look for the best reply, in seconds (default: %(default)s)",
    )
    parser.add_argument("-T", "--train", metavar="<file>", help="train the brain with a text file before chatting")
    args = parser.parse_args(argv)

    with MegaHAL(brainfile=args.brainfile, order=args.order, timeout=args.timeout) as megahal:
        if args.train:
            megahal.train(args.train)
        megahal.interact()

    return 0


if __name__ == "__main__":
    sys.exit(main())
