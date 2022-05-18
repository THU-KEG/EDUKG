# -*- coding: UTF-8 -*-
from ltp import LTP
import re
import os


class EntityHandler:
    def __init__(self, word_dict=None):
        self.ltp = LTP()
        self.wd_path = word_dict
        self.illegal_char = ''
        self.word_dict = {
            'concept': [],
            'rhe_des': [],
            'rhe_def': [],
            'rhe_proc': [],
            'rhe_mec': [],
            'rhe_ill': [],
            'rhe_rel': [],
            'rhe_cause': [],
            'rhe_eff': [],
            'rhe_sig': []
        }
        self.forbidden = []
        if self.wd_path is not None:
            with open(self.wd_path, 'r') as config:
                lines = config.readlines()
                for line in lines:
                    keywords = line.strip('\n').split()
                    if keywords[0] == 'illegal_character':
                        self.illegal_char = keywords[1]
                    elif keywords[0] == 'forbidden':
                        self.forbidden = keywords[1:]
                    else:
                        if keywords[0] not in self.word_dict.keys():
                            self.word_dict[keywords[0]] = []
                        self.word_dict[keywords[0]].extend(keywords[1:])

    def st_parsing(self, label: [str]):
        try:
            for word in self.forbidden:
                if word in label[0]:
                    return None, None, None
            seg, hidden = self.ltp.seg(label)
            return seg, self.ltp.pos(hidden), self.ltp.sdp(hidden, mode='tree')
        except RuntimeError:
            return None, None, None

    def classify(self, x: [str]):
        clses = self.parse(x)
        if clses is None:
            return 'garbage'
        else:
            for i in clses.keys():
                if clses[i].is_root():
                    if clses[i].get_depth() == 1:
                        return 'concept'
                    elif clses[i].get_depth() == 2:
                        if 'eCOO' in [tup[0] for tup in clses[i].get_lower()]:
                            return 'rhe_rel'
                        else:
                            return 'concept'
                    else:
                        if 'eCOO' in [tup[0] for tup in clses[i].get_lower()]:
                            return 'rhe_rel'
                        for category in self.word_dict:
                            for key in self.word_dict[category]:
                                if key in x[0]:
                                    return category
                        return 'rhe_des'

    def classify_ch(self, x: str):
        if x is None:
            return 'garbage'
        else:
            for category in self.word_dict:
                for key in self.word_dict[category]:
                    if key in x:
                        return category
            return 'concept'

    def classify_math(self, x: [str]):
        clses = self.parse(x)
        if clses is None:
            return 'garbage'
        else:
            for i in clses.keys():
                if clses[i].is_root():
                    if clses[i].get_depth() == 1:
                        return 'concept'
                    else:
                        if 'eCOO' in [tup[0] for tup in clses[i].get_lower()]:
                            return 'rhe_rel'
                        else:
                            if '的' in x[0]:
                                for category in self.word_dict:
                                    for key in self.word_dict[category]:
                                        if key in x[0]:
                                            return category
                                return 'rhe_des'
                            else:
                                return 'concept'

    def parse(self, label: [str]):

        def has_illegal(label: str):
            for char in self.illegal_char:
                if char in label:
                    return True
            return False

        x = [re.sub('\（.*\）', '', label[0])]
        if has_illegal(x[0]) or x[0].endswith('是'):
            return None
        else:
            seg, pos, sdp = self.st_parsing(x)
            if seg is None:
                return None
            components = {}
            for i in range(len(seg[0])):
                components[i] = Component(seg[0][i], pos[0][i])

            for i in range(len(sdp[0])):
                if sdp[0][i][2] == 'Root':
                    components[sdp[0][i][0] - 1].set_as_root()
                else:
                    try:
                        components[sdp[0][i][0] - 1].add_upper(sdp[0][i][2], components[sdp[0][i][1] - 1])
                        components[sdp[0][i][1] - 1].add_lower(sdp[0][i][2], components[sdp[0][i][0] - 1])
                    except KeyError:
                        return None

            return components


class Component:
    def __init__(self, seg, pos):
        self.upper = []
        self.lower = []
        self.root = False
        self.seg = seg
        self.pos = pos

    def add_upper(self, r, t):
        self.upper.append((r, t))

    def add_lower(self, r, t):
        self.lower.append((r, t))

    def set_as_root(self):
        self.root = True

    def get_upper(self):
        return self.upper

    def get_lower(self):
        return self.lower

    def is_root(self):
        return self.root

    def get_depth(self):
        cur_depth = 0
        for lower in self.lower:
            if lower[1].get_depth() > cur_depth:
                cur_depth = lower[1].get_depth()
        cur_depth += 1
        return cur_depth

    def get_seg(self):
        return self.seg
