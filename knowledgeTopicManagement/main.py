import time

from entity_handler import EntityHandler
from kg_handler import KGHandler
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal
import pandas as pd
import os
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sparql_query import JenaConnector
from sentence_transformers import SentenceTransformer
import numpy as np
from utility import cosine_vecbyvec
import torch
from disambigua import *
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick


XLORE_REQUEST = 'http://api.xlore.cn/query?instances='


class Manager:
    def __init__(self, src_root='./data/src/', tar_root='./data/tar/'):
        self.kg_handler = KGHandler(f'{src_root}edukg3_ttl_latest/', keys=
        ['chemistry', 'chinese', 'history', 'geo', 'biology', 'politics', 'physics', 'math'])
        self.ent_handler = EntityHandler()
        self.src_root = src_root
        self.tar_root = tar_root

    def clean(self, subject, to_csv=False):
        assert subject in self.kg_handler.graphs.keys()

        g = self.kg_handler.graphs[subject]
        dfs = self.long_short_typing(subject, to_csv=to_csv)
        print(dfs.keys())
        if 'garbage' not in dfs.keys():
            return None

        uris = dfs['garbage']['uri'].values
        print(len(uris))

        for h, r, t in g:
            if h.n3() in uris or t.n3() in uris:
                g.remove((h, r, t))

        return g

    def long_short_typing(self, subject, to_csv=False):
        assert subject in self.kg_handler.graphs.keys()
        entities = self.kg_handler._get_entities_label(subject)

        if os.path.isfile(f'./data/{subject}_classification_key.txt'):
            self.ent_handler = EntityHandler(f'./data/{subject}_classification_key.txt')

        labels = {}

        if subject == 'chinese':
            for uri in tqdm(entities.keys()):
                tp = self.ent_handler.classify_ch(entities[uri]['label'])
                if tp not in labels.keys():
                    labels[tp] = []
                labels[tp].append(uri)
        elif subject == 'math':
            for uri in tqdm(entities.keys()):
                x = [entities[uri]['label']]
                try:
                    tp = self.ent_handler.classify_math(x)
                except AttributeError:
                    continue
                if tp not in labels.keys():
                    labels[tp] = []
                labels[tp].append(uri)

        else:
            for uri in tqdm(entities.keys()):
                x = [entities[uri]['label']]
                try:
                    tp = self.ent_handler.classify(x)
                except AttributeError:
                    continue
                if tp not in labels.keys():
                    labels[tp] = []
                labels[tp].append(uri)

        dfs = {}
        for key in labels.keys():
            data = []
            attr_list = list(entities.values())[0].keys()
            for uri in labels[key]:
                data_list = [uri]
                for attr in attr_list:
                    data_list.append(entities[uri][attr])
                data.append(data_list)
            columns = ['uri']
            columns.extend(attr_list)
            df = pd.DataFrame(data, columns=columns)
            dfs[key] = df

        if to_csv:
            if not os.path.isdir(self.tar_root+'long_short'):
                os.mkdir(self.tar_root+'long_short')
            if not os.path.isdir(f'{self.tar_root}long_short/{subject}'):
                os.mkdir(f'{self.tar_root}long_short/{subject}')
            for key in dfs.keys():
                dfs[key].to_csv(f'{self.tar_root}long_short/{subject}/{key}_entities.csv', encoding='utf_8_sig', index=False)

        return dfs

    def print_class_entities(self, idxs, subj, n=30):
        res = self.kg_handler.get_class_entities(idxs, subj, n)

        if subj == 'chinese':
            for r in res:
                if self.ent_handler.classify_ch(r) == 'concept':
                    print(r)

        elif subj == 'math':
            for r in res:
                if self.ent_handler.classify_math([r]) == 'concept':
                    print(r)
        else:
            for r in res:
                if self.ent_handler.classify([r]) == 'concept':
                    print(r)

    def get_degrees(self, in_out=True):
        concept_in_d, concept_out_d = {}, {}
        rhe_in_d, rhe_out_d = {}, {}
        rhe, conc = {}, {}
        subjects = ['biology', 'chemistry', 'chinese', 'geo', 'history', 'math', 'physics', 'politics']

        if in_out:
            for subject in subjects:
                concepts = pd.read_csv(
                    f'./data/tar/long_short/{subject}/concept_entities.csv', encoding='utf-8')['uri'].tolist()

                for h, r, t in self.kg_handler.graphs[subject]:
                    head = h.n3()
                    tail = t.n3()
                    if head in concepts:
                        if head not in concept_out_d.keys():
                            concept_out_d[head] = 0
                        concept_out_d[head] += 1
                    if tail in concepts:
                        if tail not in concept_in_d.keys():
                            concept_in_d[tail] = 0
                        concept_in_d[tail] += 1

                for h, r, t in self.kg_handler.graphs[subject]:
                    head = h.n3()
                    tail = t.n3()
                    if head not in concepts:
                        if head not in rhe_out_d.keys():
                            rhe_out_d[head] = 0
                        rhe_out_d[head] += 1
                    if tail not in concepts and isinstance(t, URIRef):
                        if tail not in rhe_in_d.keys():
                            rhe_in_d[tail] = 0
                        rhe_in_d[tail] += 1

            return concept_in_d, concept_out_d, rhe_in_d, rhe_out_d

        else:
            for subject in subjects:
                concepts = pd.read_csv(
                    f'./data/tar/long_short/{subject}/concept_entities.csv', encoding='utf-8')['uri'].tolist()

                for h, r, t in self.kg_handler.graphs[subject]:
                    head = h.n3()
                    tail = t.n3()
                    if head in concepts:
                        if head not in conc.keys():
                            conc[head] = 0
                        conc[head] += 1
                    if tail in concepts:
                        if tail not in conc.keys():
                            conc[tail] = 0
                        conc[tail] += 1

                for h, r, t in self.kg_handler.graphs[subject]:
                    head = h.n3()
                    tail = t.n3()
                    if head not in concepts:
                        if head not in rhe.keys():
                            rhe[head] = 0
                        rhe[head] += 1
                    if tail not in concepts and isinstance(t, URIRef):
                        if tail not in rhe.keys():
                            rhe[tail] = 0
                        rhe[tail] += 1

            return conc, rhe

    def disambiguate(self, uri, name, subject, model):
        ctx = self.kg_handler.get_context(URIRef(uri), subject)
        ctx_emb = model.encode(ctx)
        try:
            candidates = requests.get(XLORE_REQUEST + name).json()['Instances']
        except requests.exceptions.ConnectionError:
            time.sleep(60)
            candidates = requests.get(XLORE_REQUEST + name).json()['Instances']
        if len(candidates) == 0:
            return ''
        candidates = {candidate['Uri']: candidate['Abstracts'].strip()[:15] for candidate in candidates}
        cand_emb = model.encode(list(candidates.values()))
        top = np.argmax([cosine_vecbyvec(vec, ctx_emb) for vec in cand_emb])
        result = list(candidates.keys())[top]
        return result

    def concept_xlore_linking(self, to_csv=True):
        files_path = './data/tar/long_short/'
        subjects = os.listdir(files_path)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for subject in subjects:
            assert subject in self.kg_handler.graphs.keys()
            if subject != 'chinese':
                df = pd.read_csv('{}{}/concept_entities.csv'.format(files_path, subject), encoding='utf_8_sig')
                if 'xlore' not in df.columns:
                    df['xlore'] = ''
                    print('linking {}'.format(subject))
                    for i in tqdm(df.index):
                        uri = df.loc[i]['uri'].replace('<', '').replace('>', '')
                        label = df.loc[i]['label']
                        df.loc[i]['xlore'] = self.disambiguate(uri, label, subject, model)
                    if to_csv:
                        df.to_csv('{}{}/concept_entities.csv'.format(files_path, subject), encoding='utf_8_sig')

    def main(self):
        """concept_in_d, concept_out_d, rhe_in_d, rhe_out_d = self.get_degrees()
        data = []

        for key in concept_in_d.keys():
            if key in concept_out_d.keys():
                data.append((concept_in_d[key], concept_out_d[key], 'Concept'))
            else:
                data.append((concept_in_d[key], 0, 'Concept'))

        for key in concept_out_d.keys():
            if key not in concept_in_d.keys():
                data.append((0, concept_out_d[key], 'Concept'))

        for key in rhe_in_d.keys():
            if key in rhe_out_d.keys():
                data.append((rhe_in_d[key], rhe_out_d[key], 'Rhetorical Role'))
            else:
                data.append((rhe_in_d[key], 0, 'Rhetorical Role'))

        for key in rhe_out_d.keys():
            if key not in rhe_in_d.keys():
                data.append((0, rhe_out_d[key], 'Rhetorical Role'))

        df = pd.DataFrame(data, columns=['In Degree', 'Out Degree', 'hue'])
        df.to_csv('./sns_data.csv', encoding='utf-8', index=False)

        print(df.head())

        sns.jointplot(df, x='In Degree', y='Out Degree', hue='hue', kind="hex", color="#4CB391")
        plt.show()"""

        fig, ax = plt.subplots(figsize=(12, 10))

        conc, rhe = self.get_degrees(in_out=False)
        conc = [x for x in list(conc.values()) if x < 20]
        rhe = [x for x in list(rhe.values()) if x < 20]
        df_conc = pd.DataFrame(conc, columns=['degree'])
        df_rhe = pd.DataFrame(rhe, columns=['degree'])
        df_conc.to_csv('./conc.csv')
        df_rhe.to_csv('./rhe.csv')

        sns.distplot(conc)
        sns.distplot(rhe)
        plt.xlabel('Degree')
        plt.ylabel('Density')
        plt.show()


if __name__ == '__main__':
    # manager = Manager()
    # manager.main()
    plt.rc('font', size=26)
    sns.set_theme(style="ticks")

    fig, ax = plt.subplots(figsize=(24, 20))
    df_conc = pd.read_csv('./conc.csv', encoding='utf-8', index_col=0)
    df_conc['Knowledge Topic Type'] = 'Concept'
    df_rhe = pd.read_csv('./rhe.csv', encoding='utf-8', index_col=0)
    df_rhe['Knowledge Topic Type'] = 'Rhetorical Role'

    df = pd.DataFrame(columns=['degree', 'Knowledge Topic Type'])

    # df = pd.concat((df, df_conc, df_rhe), ignore_index=True)

    sns.set(style="ticks")
    plt.grid()
    """p = sns.histplot(data=df, x='degree', hue='Knowledge Topic Type', multiple="dodge", bins=10, shrink=.8,
                     kde=True, line_kws={"lw": 5, 'ls': '-'}, alpha=0.9)"""
    sns.kdeplot(x=df_conc['degree'], lw=5, ls='-')
    sns.kdeplot(x=df_rhe['degree'], lw=5, ls='--')

    plt.xlabel('Degree', fontsize=52)
    plt.ylabel('Percentage', fontsize=52, loc='center')
    plt.xticks(fontsize=36)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.yticks(fontsize=36)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0%}'.format(x)))
    plt.legend(labels=["Concept", "Rhetorical Role"], fontsize=42)
    # plt.setp(p.get_legend().get_texts(), fontsize='52')
    # plt.setp(p.get_legend().get_title(), fontsize='0')

    plt.show()

