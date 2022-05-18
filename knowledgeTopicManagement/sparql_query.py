from numpy.lib.nanfunctions import nanmean
import requests
import re
import os
import traceback
import json
from bidict import bidict
from py2neo import Graph as NeoGraph
import pandas as pd
from tqdm import tqdm
import zhconv
from itertools import product, combinations
from queue import Queue
import networkx as nx
from bs4 import BeautifulSoup
import pandas as pd
from pypinyin import lazy_pinyin
import random
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON


def get_edukg_triplets(graph_iri):
    sparql = SPARQLWrapper("http://39.97.172.123:8890/sparql")
    sparql.setQuery("select count(*) from <%s> where { ?s ?p ?o .}" % graph_iri)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    triplet_cnt = int(result['results']['bindings'][0]['callret-0']['value'])
    limit = (int(triplet_cnt / 10000) + 1) * 10000
    query_template = "select * from <%s> where { ?s ?p ?o .} LIMIT %d OFFSET %d"
    results = []
    for offset in tqdm(range(0, limit, 10000)):
        sparql.setQuery(query_template % (graph_iri, limit, offset))
        sparql.setReturnFormat(JSON)
        results += [(triple['s']['value'], triple['p']['value'], triple['o']['value']) for triple in
                    sparql.query().convert()['results']['bindings']]
    return pd.DataFrame(results, columns=['s', 'p', 'o'])


def fetch_edukg2_data():
    results = {
        course: get_edukg_triplets("http://edukg.org/%s" % course)
        for course in COURSES_EN
    }
    for course in results:
        results[course].to_excel('data/edukg2.0/edukg2_ttl/new_main/%s_main.xlsx' % course, index=False)


class HistoricQueue(Queue):
    def __init__(self, maxsize=0):
        Queue.__init__(self, maxsize=maxsize)
        self.history_set = set()

    def _put(self, item):
        if item not in self.history_set:
            Queue._put(self, item)
            self.history_set.add(item)

    def _get(self):
        return Queue._get(self)


from utility import remove_puncts, TYPE_PREDICATE, LABEL_PREDICATE, CLASS_PREDICATE, COURSES_ZH, COURSES_EN, TO_ZH_TR, \
    TO_ZH_SI, NAMESPACE2URI

EDUKG_CLASSES = pd.read_excel('data/edukg-class.xlsx', engine='openpyxl')
EDUKG_NAME2URI = {}
EDUKG_URI2NAME = {}
for i in range(EDUKG_CLASSES.shape[0]):
    uri = EDUKG_CLASSES.iloc[i]['uri']
    name = EDUKG_CLASSES.iloc[i]['name']
    EDUKG_URI2NAME[uri] = name
    if name in EDUKG_NAME2URI:
        EDUKG_NAME2URI[name].append(uri)
    else:
        EDUKG_NAME2URI[name] = [uri]

PREFIXES = bidict({
    "rdf": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
    "rdfs": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
    "owl": "PREFIX owl: <http://www.w3.org/2002/07/owl#>",
    "xsd": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
    "xlore_ins": "PREFIX ins: <http://xlore.org/instance/>",
    "xlore_cls": "PREFIX cls: <http://xlore.org/concept/>",
    "xlore_prop": "PREFIX prop: <http://xlore.org/property/>"
})

# There are four slots in the Sparql query template, which are prefixes, variables, conditions and limitations, respectively
SPARQL_TEMPLATE = '''
    %s
    SELECT DISTINCT %s
    WHERE{
        %s
    }
    %s
'''

SUPERCLASS_QUERY = '''
    %s
    SELECT DISTINCT ?class ?label
    WHERE{
        ?child rdfs:label "Category:%s"@zh .
        ?child owl:SubClassOf ?class .
        ?class rdfs:label ?label.
    }
'''

SUBCLASS_QUERY = '''
    %s
    SELECT DISTINCT ?class ?label
    WHERE {
        ?class a owl:Class.
        ?class owl:SubClassOf ?father.
        ?father rdfs:label "Category:%s"@zh.
        OPTIONAL { ?class rdfs:label ?label}
    }
'''

SUBSTRING_FILTER = '''
    FILTER(
        contains (str(?class), "%s")
    )
'''

CYPHER_SIMPLE_TEMPLATE = 'MATCH (n:`%s`) RETURN n'
GET_CLASSES = lambda x: [{'uri': v['object']['value'], 'name': v['oname']['value'] if 'oname' in v else None} for v in
                         x['results']['bindings'] if
                         v['predicate']['value'] == CLASS_PREDICATE] if x is not None else []

MODE2FUNC = {
    'both': [TO_ZH_TR, TO_ZH_SI],
    'zh-si': [TO_ZH_SI],
    'zh-tr': [TO_ZH_TR],
}


class SPARQL():
    def __init__(
            self,
            prefixes=PREFIXES,
            variables=["name"],
            conditions=[["name", "rdfs:label", "\"Category:人物\"@zh"]],
            limitations=25,
            filters=[]
    ):
        self.prefixes = prefixes
        self.variables = variables
        self.conditions = conditions
        self.limitations = limitations
        self.query_str = None
        self.filters = filters

    def _parse(self, v):
        if not isinstance(v, str):
            v = str(v)
        if v.startswith('http'):
            v = '<' + v + '>'
            if v in [val.split(' ')[-1] for val in self.prefixes.values()]:
                v = self.prefixes.inv[v]
            return v
        elif '"' in v or re.search('.+:.+', v):
            return v
        else:
            return '?' + v

    def get_str(self):
        conditions = []
        for c in self.conditions:
            if len(c) == 3:
                new_cond = ' '.join([self._parse(v) for v in c]) + '.'
            else:
                new_cond = '%s {%s}' % (c[0], ' '.join([self._parse(v) for v in c[1:]]))
            conditions.append(new_cond)
        for f in self.filters:
            conditions.append('FILTER(%s(%s))' % (f[0], ', '.join([self._parse(v) for v in f[1:]])))
        conditions_str = '\n'.join(conditions)
        limit = ''
        if self.limitations is not None:
            if isinstance(self.limitations, int):
                limit = 'LIMIT ' + str(self.limitations)
        self.query_str = SPARQL_TEMPLATE % (
            '\n'.join(list(self.prefixes.values())),
            ' '.join(['?' + v for v in self.variables]),
            conditions_str,
            limit
        )
        return self.query_str


class JenaConnector():
    def __init__(self, ip="10.1.1.39", port=3030, database="XLoreBaidu", in_memory_db=None):
        self.ip = ip
        self.port = port
        self.database = database
        self.url = "http://%s:%d/%s/query" % (ip, port, database)
        self.cache = {}
        self.g = in_memory_db if in_memory_db is not None and isinstance(in_memory_db, rdflib.Graph) else None

    def _query(self, query_str, force_request=False):
        try:
            query_str = query_str.replace(u"\uFFFD", '')
            if query_str in self.cache and not force_request:
                return self.cache[query_str]
            if self.g is not None:
                return self.g.query(query_str)
            else:
                r = requests.post(self.url, data={'query': query_str}, timeout=120)
                if r.status_code == 200:
                    result = r.json()
                    self.cache[query_str] = result
                    return result
                else:
                    print('Response error with status code %d.' % r.status_code)
                    return None
        except Exception as e:
            traceback.print_exc()
            return {
                'results': {
                    'bindings': []
                }
            }

    def _merge_result(self, results):
        results = [res for res in results if res is not None]
        if len(results) == 0:
            return {
                'head': {
                    'vars': []
                },
                'results': {
                    'bindings': []
                }
            }
        else:
            return {
                'head': {
                    'vars': list(set([
                        v for res in results for v in res['head']['vars']
                    ]))
                },
                'results': {
                    'bindings': [
                        v for res in results for v in res['results']['bindings']
                    ]
                }
            }

    def query_instance(self, uri):
        query_str = SPARQL(
            variables=['predicate', 'object', 'pname', 'oname'],
            conditions=[
                [uri, 'predicate', 'object'],
                ['OPTIONAL', 'predicate', 'rdfs:label', 'pname'],
                ['OPTIONAL', 'object', 'rdfs:label', 'oname']
            ],
            limitations=None
        ).get_str()
        return self._query(query_str)

    def _query_instance_by_name(self, name, base):
        query_str = SPARQL(
            variables=['subject', 'predicate', 'object', 'pname', 'oname'],
            conditions=[
                ['subject', 'rdfs:label', '"%s"@%s' % (name, base)],
                ['subject', 'predicate', 'object'],
                ['OPTIONAL', 'predicate', 'rdfs:label', 'pname'],
                ['OPTIONAL', 'object', 'rdfs:label', 'oname']
            ],
            limitations=None
        ).get_str()
        return self._query(query_str)

    def _query_uri_by_name(self, name, base):
        query_str = SPARQL(
            variables=['subject'],
            conditions=[
                ['subject', 'rdfs:label', '"%s"@%s' % (name, base)],
            ],
            limitations=None
        ).get_str()
        return self._query(query_str)

    def _query_uri_by_name_fuzzy(self, name):
        query_str = SPARQL(
            variables=['subject', 'label'],
            conditions=[
                ['subject', 'rdfs:label', 'label'],
            ],
            limitations=None,
            filters=[['contains', 'label', '"%s"' % name]]
        ).get_str()
        return self._query(query_str)

    def _query_sameas_by_uri(self, uri):
        query_str = SPARQL(
            variables=['object'],
            conditions=[
                [uri, 'owl:sameAs', 'object'],
            ],
            limitations=None,
        ).get_str()
        return self._query(query_str)

    def query_instance_by_name(self, name, mode='zh-si', base='bd'):
        # Querying with a chinese name of an instance from the Apache Jena Database (XLore)
        # @parameter:mode can be 'both', 'zh-tr' or 'zh-si';
        # @parameter:base can be 'all', 'bd', 'zh' or 'en'
        # name = remove_puncts(name)
        MODE2FUNC = {
            'both': [TO_ZH_TR, TO_ZH_SI],
            'zh-si': [TO_ZH_SI],
            'zh-tr': [TO_ZH_TR],
        }
        if mode not in MODE2FUNC:
            raise BaseException("The query word 'mode' should be in ['both', 'zh-tr', 'zh-si']")
        if base not in ['zh', 'bd', 'en', 'all']:
            raise BaseException("The query word 'base' should be in ['zh', 'bd', 'en'. 'all']")
        mode_val = MODE2FUNC[mode]
        base_val = ['zh', 'bd', 'en'] if base == 'all' else [base]
        choices = product(mode_val, base_val)
        result_list = [self._query_instance_by_name(func(name), b) for func, b in choices]
        result = result_list[0] if len(result_list) == 1 else self._merge_result(result_list)
        return {
            'results': result['results']['bindings'],
            'uri': result['results']['bindings'][0]['subject']['value'] if len(result['results']['bindings']) else None
        }

    def query_class_instances(self, uri):
        if isinstance(uri, str):
            query_str = SPARQL(
                variables=['subject', 'sname'],
                conditions=[
                    ['subject', 'owl:InstanceOf', uri],
                    ['OPTIONAL', 'subject', 'rdfs:label', 'sname'],
                ],
                limitations=None
            ).get_str()
        elif isinstance(uri, list):
            query_str = SPARQL(
                variables=['subject', 'sname'],
                conditions=[['subject', 'owl:InstanceOf', u] for u in uri]
                           + [['OPTIONAL', 'subject', 'rdfs:label', 'sname']],
                limitations=None
            ).get_str()
        return self._query(query_str)

    def query_class_fathers(self, class_uri):
        query_str = SPARQL(
            variables=['father', 'fathername'],
            conditions=[
                [class_uri, 'owl:SubClassOf', 'father'],
                ['OPTIONAL', 'father', 'rdfs:label', 'fathername'],
            ],
            limitations=None
        ).get_str()
        return self._query(query_str)

    def query_class_children(self, class_uri):
        query_str = SPARQL(
            variables=['children', 'childrenname'],
            conditions=[
                ['children', 'owl:SubClassOf', class_uri],
                ['OPTIONAL', 'children', 'rdfs:label', 'childrenname'],
            ],
            limitations=None
        ).get_str()
        return self._query(query_str)

    def find_same_entities(self, uri, search_layer=1):
        result = {}
        q = HistoricQueue()
        q.put(uri)
        for i in range(search_layer + 1):
            new_instances = []
            while not q.empty():
                this_uri = q.get()
                query_result = self.query_instance(this_uri)
                ouri2name = {v['object']: v['oname'] for v in query_result if 'oname' in v}
                class_uris = [v['uri'] for v in GET_CLASSES(query_result)]
                print(class_uris)
                for class_uri in class_uris:
                    siblings = self.query_class_instances(class_uri)['results']['bindings']
                    result[class_uri] = {
                        'layer': i + 1,
                        'name': ouri2name[class_uri] if class_uri in ouri2name else class_uri,
                        'instances': siblings
                    }
                    new_instances += [sib['subject']['value'] for sib in siblings]
            for ins in new_instances:
                q.put(ins)
        return result

    def find_subtree_instances(self, uri, search_layer=5, max_instances=1000):
        result = {}
        q = HistoricQueue()
        q.put(uri)
        node_name, nodes, edges = {}, [], []
        nodes.append(uri)
        node_name[uri] = self.get_name(uri)
        for i in range(search_layer + 1):
            new_classes = []
            while not q.empty():
                this_class = q.get()
                result[this_class] = {
                    'layer': i + 1,
                }
                children = self.query_class_children(this_class)
                for c in children['results']['bindings']:
                    new_classes.append(c['children']['value'])
                    childrenname = c['childrenname']['value'] if 'childrenname' in c else None
                    nodes.append(c['children']['value'])
                    node_name[c['children']['value']] = childrenname
                    edges.append((c['children']['value'], this_class))
            for c in new_classes:
                q.put(c)
        g = nx.DiGraph()
        nodes = list(set(nodes))
        edges = list(set(edges))
        g.add_nodes_from(nodes)
        for n in node_name:
            g.nodes[n]['name'] = node_name[n]
        g.add_edges_from(edges)
        for c in result:
            bindings = self.query_class_instances(c)['results']['bindings']
            result[c]['instances'] = random.sample(bindings, max_instances) if len(
                bindings) > max_instances else bindings
        return g, result

    def find_same_entities_by_class(self, uri, search_layer=1, max_instances=1000):
        result = {}
        q = HistoricQueue()
        query_result = self.query_instance(uri)
        classes = GET_CLASSES(query_result)
        node_name, nodes, edges = {}, [], []
        for c in classes:
            q.put(c['uri'])
            nodes.append(c['uri'])
            node_name[c['uri']] = c['name']
        for i in range(search_layer + 1):
            new_classes = []
            while not q.empty():
                this_class = q.get()
                result[this_class] = {
                    'layer': i + 1,
                }
                fathers = self.query_class_fathers(this_class)
                children = self.query_class_children(this_class)
                for f in fathers['results']['bindings']:
                    new_classes.append(f['father']['value'])
                    fathername = f['fathername']['value'] if 'fathername' in f else None
                    nodes.append(f['father']['value'])
                    node_name[f['father']['value']] = fathername
                    edges.append((this_class, f['father']['value']))
                for c in children['results']['bindings']:
                    new_classes.append(c['children']['value'])
                    childrenname = c['childrenname']['value'] if 'childrenname' in c else None
                    nodes.append(c['children']['value'])
                    node_name[c['children']['value']] = childrenname
                    edges.append((c['children']['value'], this_class))
            for c in new_classes:
                q.put(c)

        g = nx.DiGraph()
        nodes = list(set(nodes))
        edges = list(set(edges))
        g.add_nodes_from(nodes)
        for n in node_name:
            g.nodes[n]['name'] = node_name[n]
        g.add_edges_from(edges)
        for c in result:
            bindings = self.query_class_instances(c)['results']['bindings']
            result[c]['instances'] = random.sample(bindings, max_instances) if len(
                bindings) > max_instances else bindings
        return g, result

    def find_entity_by_concept_combos(self, concepts, select=2, max_instances=500):
        result = {}
        combos = list(combinations(concepts, select))
        for combo in combos:
            bindings = self.query_class_instances(list(combo))['results']['bindings']
            result[combo] = random.sample(bindings, max_instances) if len(bindings) > max_instances else bindings
        return result

    def get_subclass_network(self):
        query_str = SPARQL(
            variables=['children', 'father', 'childrenname', 'fathername'],
            conditions=[
                ['children', 'owl:SubClassOf', 'father'],
                ['OPTIONAL', 'children', 'rdfs:label', 'childrenname'],
                ['OPTIONAL', 'father', 'rdfs:label', 'fathername'],
            ],
            limitations=None
        ).get_str()
        query_result = self._query(query_str)['results']['bindings']
        node_name, nodes, edges = {}, [], []
        for item in query_result:
            child, father = item['children']['value'], item['father']['value']
            if 'childrenname' in item:
                node_name[child] = item['childrenname']['value']
            if 'fathername' in item:
                node_name[father] = item['fathername']['value']
            nodes.append(child)
            nodes.append(father)
            edges.append((child, father))
        nodes = list(set(nodes))
        edges = list(set(edges))
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        for n in node_name:
            g.nodes[n]['name'] = node_name[n]
        g.add_edges_from(edges)
        return g

    def get_name(self, uri):
        query_str = SPARQL(
            variables=['name'],
            conditions=[
                [uri, 'rdfs:label', 'name'],
            ],
            limitations=None
        ).get_str()
        result = self._query(query_str)
        if result is not None and len(result['results']['bindings']):
            return result['results']['bindings'][0]['name']['value']
        else:
            return None

    def get_json(self, uri):
        assert isinstance(uri, str)
        query_res = self.query_instance(uri)['results']['bindings']
        predicate2name = {v['predicate']['value']: v['pname']['value']
                          for v in query_res if 'pname' in v
                          }
        if len(query_res) == 0:
            return None
        triplet_dict = {}
        result = json.load(open('data/entity-template.json', 'r'))
        for r in query_res:
            obj_item = (r['object']['value'], r['oname']['value'] if 'oname' in r else None)
            if r['predicate']['value'] in triplet_dict:
                triplet_dict[r['predicate']['value']].append(obj_item)
            else:
                triplet_dict[r['predicate']['value']] = [obj_item]
        result['uri'] = uri
        result['label'] = [v[0] for v in triplet_dict[LABEL_PREDICATE]]
        result['polysemys'] = triplet_dict['http://xlore.org/property/supplement'][0][
            0] if 'http://xlore.org/property/supplement' in triplet_dict else ''
        if CLASS_PREDICATE in triplet_dict:
            result['superClasses']['total'] = len(triplet_dict[CLASS_PREDICATE])
            class_by_initial = {}
            for v in triplet_dict[CLASS_PREDICATE]:
                initial = lazy_pinyin(v[1])[0][0].upper()
                if initial not in class_by_initial:
                    class_by_initial[initial] = {
                        'innerstr': [v[1]],
                        'uriList': [v[0]],
                    }
                else:
                    class_by_initial[initial]['innerstr'].append(v[1])
                    class_by_initial[initial]['uriList'].append(v[0])
            result['superClasses']['linkModelArray'] = [{
                'initial': k,
                'innerstr': class_by_initial[k]['innerstr'],
                'uriList': class_by_initial[k]['uriList'],
            } for k in class_by_initial]
        result['abstracts'] = triplet_dict['http://www.w3.org/2000/01/rdf-schema#comment'][0][
            0] if 'http://www.w3.org/2000/01/rdf-schema#comment' in triplet_dict else ""
        infobox = {
            k: triplet_dict[k] for k in triplet_dict if
            'comment' not in k and 'isRelatedTo' not in k and 'property' in k and 'hasURL' not in k and 'supplement' not in k
        }
        for k in infobox:
            result['infobox'].append({
                'propUri': k,
                'label': predicate2name[k],
                'insUri': {
                    val[1] if val[1] is not None else val[0]: val[0] if val[1] is not None else ""
                    for val in infobox[k]
                },
                'value': [v[1] if v[1] is not None else v[0] for v in infobox[k]]
            })
        return result

    def get_json_by_name(self, name, base='bd', mode='zh-si'):
        # name = remove_puncts(name)
        if mode not in MODE2FUNC:
            raise BaseException("The query word 'mode' should be in ['both', 'zh-tr', 'zh-si']")
        if base not in ['zh', 'bd', 'en', 'all']:
            raise BaseException("The query word 'base' should be in ['zh', 'bd', 'en'. 'all']")
        mode_val = MODE2FUNC[mode]
        base_val = ['zh', 'bd', 'en'] if base == 'all' else [base]
        choices = product(mode_val, base_val)
        uris = self._merge_result([self._query_uri_by_name(func(name), b) for func, b in choices])
        uris = list(set([v['subject']['value'] for v in uris['results']['bindings']]))
        return {
            uri: self.get_json(uri)
            for uri in uris
        }

    def get_comment(self, uri):
        query_str = SPARQL(
            variables=['object'],
            conditions=[
                [uri, 'rdfs:comment', 'object'],
            ],
            limitations=None
        ).get_str()
        query_result = self._query(query_str)
        if query_result is not None:
            return '\n'.join([b['object']['value'] for b in query_result['results']['bindings']])
        else:
            return ''

    def get_class(self, uri):
        query_str = SPARQL(
            variables=['object'],
            conditions=[
                [uri, 'owl:InstanceOf', 'object'],
            ],
            limitations=None
        ).get_str()
        query_result = self._query(query_str)
        if query_result is not None:
            return [b['object']['value'] for b in query_result['results']['bindings']]
        else:
            return []

    def get_uri_by_name(self, name, keep_accurate=True, keep_partial=0.5, fuzzy=False, base='bd', mode='zh-si'):
        if fuzzy:
            query_result = self._query_uri_by_name_fuzzy(name)
            if query_result is not None:
                sorted_result = sorted(query_result['results']['bindings'], key=lambda x: len(x['label']['value']))
                if not keep_accurate:
                    return [v['subject']['value'] for v in sorted_result], all(
                        [v['label']['value'] == name for v in sorted_result])
                else:
                    if sorted_result[0]['label']['value'] == name:
                        return [v['subject']['value'] for v in sorted_result if v['label']['value'] == name], True
                    else:
                        return [v['subject']['value'] for v in
                                sorted_result[:int(len(sorted_result) * keep_partial)]], all(
                            [v['label']['value'] == name for v in sorted_result])
        else:
            if mode not in MODE2FUNC:
                raise BaseException("The query word 'mode' should be in ['both', 'zh-tr', 'zh-si']")
            if base not in ['zh', 'bd', 'en', 'all']:
                raise BaseException("The query word 'base' should be in ['zh', 'bd', 'en'. 'all']")
            mode_val = MODE2FUNC[mode]
            base_val = ['zh', 'bd', 'en'] if base == 'all' else [base]
            choices = product(mode_val, base_val)
            uris_result = self._merge_result([self._query_uri_by_name(func(name), b) for func, b in choices])
            return [v['subject']['value'] for v in uris_result['results']['bindings']], True

    def get_same_instances(self, uri):
        result = self._query_sameas_by_uri(uri)
        if result is not None:
            return [v['object']['value'] for v in result['results']['bindings']]

    def get_property_triplet(self, uri):
        query_str = SPARQL(
            variables=['sname', 'oname', 'subject', 'object'],
            conditions=[
                ['subject', uri, 'object'],
                ['OPTIONAL', 'subject', 'rdfs:label', 'sname'],
                ['OPTIONAL', 'object', 'rdfs:label', 'oname'],
            ],
            limitations=1,
            filters=[
                ['strlen', 'object']
            ]
        ).get_str()
        query_result = self._query(query_str)
        if len(query_result['results']['bindings']):
            return {k: query_result['results']['bindings'][0][k]['value'] for k in
                    query_result['results']['bindings'][0]}
        else:
            return {}

    def count_property(self, uri):
        query_str = '''
        SELECT (COUNT(DISTINCT(?subject)) as ?scount)
        WHERE {
            ?subject <%s> ?object .
        }
        ''' % uri
        query_res = self._query(query_str)
        if len(query_res['results']['bindings']) > 0:
            return int(query_res['results']['bindings'][0]['scount']['value'])
        else:
            return -1

    def get_class_descendant_children(self, root_uri, layer=1):
        if layer == 0:
            q = ''
        else:
            q = 'owl:InstanceOf/owl:SubClassOf{0,%d}' % layer
        query_str = SPARQL(
            variables=['subject', 'sname'],
            conditions=[
                ['subject', q, root_uri],
                ['subject', 'rdfs:label', 'sname'],
            ],
            limitations=None
        ).get_str()
        query_result = self._query(query_str)['results']['bindings']
        return {
            v['subject']['value']: v['sname']['value']
            for v in query_result
        }

    def get_original_url(self, uri):
        query_str = SPARQL(
            variables=['url'],
            conditions=[
                [uri, 'prop:hasURL', 'url'],
            ],
            limitations=1,
        ).get_str()
        query_result = self._query(query_str)
        if len(query_result['results']['bindings']):
            return query_result['results']['bindings'][0]['url']['value']
        else:
            return None

    def get_uri_by_baike_url(self, url):
        query_str = SPARQL(
            variables=['subject'],
            conditions=[
                ['subject', 'prop:hasURL', url],
            ],
            limitations=1,
        ).get_str()
        query_result = self._query(query_str)
        if len(query_result['results']['bindings']):
            return query_result['results']['bindings'][0]['subject']['value']
        else:
            return None


class Neo4jConnector():
    def __init__(self, ip, user, password, port=7687, scheme='bolt'):
        self.ip = ip
        self.user = user
        self.password = password
        self.port = port
        self.scheme = scheme
        self.addr = '%s://%s:%d' % (self.scheme, self.ip, self.port)
        self.g = NeoGraph(self.addr, auth=(self.user, self.password))

    def _query(self, query_str):
        try:
            res = self.g.run(query_str).data()
            return res
        except Exception as e:
            traceback.print_exc()
            return None

    def query_by_label(self, label):
        return self._query(CYPHER_SIMPLE_TEMPLATE % label)

    def add_rdf(self, file):
        self._query('CALL n10s.rdf.import.fetch("file://localhost%s", "Turtle");' % file)

    def import_rdf(self, filelist):
        self._query('MATCH (resource) DETACH DELETE resource;')
        self._query('CALL n10s.graphconfig.init();')
        for ns in NAMESPACE2URI:
            self._query(
                'CALL n10s.nsprefixes.add("%s", "%s");' % (
                    ns, str(NAMESPACE2URI[ns])
                )
            )
        self._query('''CREATE CONSTRAINT n10s_unique_uri ON (r:Resource)
        ASSERT r.uri IS UNIQUE;''')
        for file in filelist:
            self.add_rdf(file)
        return True


class EDUKGConnector():

    def __init__(self):
        headersStr = '''Host: 39.100.31.203:8001
            Proxy-Connection: keep-alive
            Content-Length: 0
            accept: */*
            User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36
            Origin: http://39.100.31.203:8001
            Referer: http://39.100.31.203:8001/swagger-ui.html
            Accept-Encoding: gzip, deflate
            Accept-Language: zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6
            Cookie: JSESSIONID=A54BBB329AE5D201405B268DFDCFBC24; WEBRSID=d37a35fc-0f7f-481a-b7e3-f73be0d9f489'''
        self.headers = {s.split(': ')[0]: s.split(': ')[1] for s in headersStr.split('\n') if s}

    def searchKG(self, subject, name):
        urlTemplate = 'http://39.100.31.203:8001/api/edukg/searchKgData?searchKey=%s&subject=%s'
        if not subject in COURSES_ZH:
            print('The input subject is not in the subjects list')
            return None
        url = urlTemplate % (name, subject)
        r = requests.post(url, headers=self.headers)
        return r.json()

    def searchText(self, subject, name, limit=200):
        urlTemplate = 'http://39.100.31.203:8001/api/edukg/searchBookData?limit=%d&searchKey=%s&subject=%s'
        if not subject in COURSES_ZH:
            print('The input subject is not in the subjects list')
            return None
        url = urlTemplate % (limit, name, subject)
        r = requests.post(url, headers=self.headers)
        return r.json()

    def searchByLabels(self, labels, fuzzy=False):
        kg_content, kg_property, txt_content = {}, {}, {}
        for label in labels:
            for subject in COURSES_ZH:
                result = self.searchKG(subject, label)
                if result['code'] == '0' and (len(result['data']['property']) or len(result['data']['content'])):
                    kg_content[subject] = result['data']['content']
                    kg_property[subject] = result['data']['property']
                limit_len = 20
                while True:
                    print('Trying with limitation %d on subject %s by the name of %s' % (limit_len, subject, label))
                    result = self.searchText(subject, label, limit=limit_len)
                    if result['code'] == '0':
                        contents = [BeautifulSoup('\n'.join(v['content'])).text if v['content'] else '' for v in
                                    result['data']]
                        if not fuzzy:
                            contents = [v for v in contents if label in v]
                        if len(contents) < limit_len:
                            txt_content[subject] = contents
                            break
                        limit_len *= 2
                    else:
                        break
        return {'kg_content': kg_content, 'kg_property': kg_property, 'txt_content': txt_content}


if __name__ == '__main__':
    uri = 'http://xlore.org/instance/zhi128214'
    connector = JenaConnector()
    res = connector.query_instance(uri)
    print(res)
    neo4j_connector = Neo4jConnector('47.94.201.245', 'neo4j', 'neo4j@keg202')
    all_result = {}
    for course in COURSES_EN:
        course_classes = EDUKG_CLASSES[EDUKG_CLASSES['uri'].apply(lambda x: course in x)]
        # Getting history related entities from Neo4j according to each history course's class
        result = {course_classes.iloc[i]['uri']: {'name': course_classes.iloc[i]['name'],
                                                  'instances': [r['n']['name'] for r in neo4j_connector.query_by_label(
                                                      course_classes.iloc[i]['uri']) if
                                                                isinstance(r['n']['name'], str) and r['n'][
                                                                    'name'] != 'Unknown']} for i in
                  tqdm(range(course_classes.shape[0]))}
        # uri_lens = sorted([(result[k]['name'], len(result[k]['instances'])) for k in result], reverse=True, key=lambda x:x[1])
        # Searching each instance in each history course's class from XLore database (Apache Jena in 10.1.1.39)
        for uri in tqdm(result):
            result[uri]['query_result'] = {}
            for k in result[uri]['instances']:
                query_result = connector.query_instance_by_name(k)
                if len(query_result['results']) > 0:
                    result[uri]['query_result'][k] = query_result
        result_df = pd.DataFrame(
            [[result[uri]['name'], ','.join(result[uri]['query_result'].keys()), len(result[uri]['query_result']), uri]
             for uri in result],
            columns=['class', 'instances', 'num', 'old-uri']
        )
        result_df.to_excel('tmp_file/%s-seed.xlsx' % course, index=False, engine='openpyxl')
        all_result[course] = result

    json.dump(all_result, open('tmp_file/seed_data.json', 'w'))
    sample = {
        course: [
            v for cls in all_result[course]
            for v in random.sample(
                all_result[course][cls]['query_result'].keys(),
                min(2,
                    len(all_result[course][cls]['query_result'].keys())
                    ))]
        for course in all_result}
    # json.dump(result, open('tmp_file/history_class_instances.json', 'w'))
    class2ins = [[result[uri]['name'], ','.join(list(result[uri]['query_result'].keys()))] for uri in result]
    class2ins_df = pd.DataFrame({
        'class': [v[0] for v in class2ins],
        'instances': [v[1] for v in class2ins],
        'uri': [EDUKG_NAME2URI[v[0]] for v in class2ins],
    })
    class2ins_df['num'] = class2ins_df['instances'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    class2ins_df.to_excel('tmp_file/history_class2ins.xlsx', engine='openpyxl', index=False)
    class_graph, new_instances = connector.find_same_entities_by_class('http://xlore.org/instance/zhi6642')
    g = connector.get_subclass_network()
    # In degree represents the number of children, and out degree represents the number of fathers

    # Fetching augmented data from zh-XLore based on manually selected type matchings
    matched_cls = pd.read_excel('tmp_file/history_matched_cls.xlsx')
    matched_cls['instances'] = matched_cls['zh-cls-uri'].progress_apply(
        lambda x: connector.get_class_descendant_children(x, layer=5))
    print(matched_cls.groupby('class')['instances'].apply(lambda x: len(x.iloc[0])).to_dict())
    matched_cls_list = matched_cls.to_dict('records')
    instance_list = [(cls_obj['class'], k, cls_obj['instances'][k]) for cls_obj in matched_cls_list for k in
                     cls_obj['instances']]
    instances_df = pd.DataFrame([
        (*v, connector.get_comment(v[1])) for v in tqdm(instance_list)
    ], columns=['y', 'uri', 'name', 'abstract'])
    instances_df.to_excel('history_class_matching_augmented_data.xlsx', index=False)
    sampled_instances = instances_df[instances_df['abstract'].apply(lambda x: len(x) > 10)].groupby('y')[
        'abstract'].apply(lambda x: random.sample(x.tolist(), min(10, len(x)))).to_dict()
    pd.DataFrame([(y, v) for y in sampled_instances for v in sampled_instances[y]], columns=['y', 'name']).to_excel(
        'tmp_file/history_class_matching_sampled_augmented_data.xlsx', index=False)

    neo4j_connector = Neo4jConnector('localhost', 'neo4j', 'neo4j@keg202')
    filelist = [os.path.join('/data/edukg_ttl', f) for f in os.listdir('/data/edukg_ttl')]
    neo4j_connector.import_rdf(filelist)