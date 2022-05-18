import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

tqdm.pandas()
import random
from itertools import product
from collections import Counter
import zhconv
import re
import os
import json
import pickle
from rdflib import Namespace, Graph, URIRef, FOAF, OWL, RDFS, RDF
import queue
from zhon import hanzi
from string import punctuation
from sklearn.model_selection import StratifiedKFold

PUNCTUATIONS = punctuation + hanzi.punctuation
remove_puncts = lambda x: re.sub(r"[%s]+" % PUNCTUATIONS, "", x)

IMPLICIT_CLASSES = ['http://www.w3.org/2002/07/owl#NamedIndividual', 'http://www.w3.org/2002/07/owl#DatatypeProperty',
                    'http://www.w3.org/2002/07/owl#Class', 'http://www.w3.org/2002/07/owl#ObjectProperty',
                    'http://www.openannotation.org/ns/Annotation', 'http://edukb.org/knowledge/0.1/class#Category',
                    'http://xmlns.com/foaf/0.1/Person']
IMPLICIT_CLASSES_NAME = ['w3-NamedIndividual', 'w3-DataProperty', 'w3-Class', 'w3-ObjectProperty',
                         'openannotation-annotation', 'edukg-Category', 'xmlns-Person']
IMPLICIT_CLASSES = {IMPLICIT_CLASSES[i]: IMPLICIT_CLASSES_NAME[i] for i in range(len(IMPLICIT_CLASSES))}
TYPE_PREDICATE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
TYPE_PREDICATE_NAME = 'w3-rdf-type'
CLASS_PREDICATE = 'http://www.w3.org/2002/07/owl#InstanceOf'
LABEL_PREDICATE = 'http://www.w3.org/2000/01/rdf-schema#label'
SUBCLASS_PREDICATE = 'http://www.w3.org/2002/07/owl#SubClassOf'

# Usage: ENTITY_TEMPALTE_NEW % (course, name, hashed-id)
ENTITY_TEMPLATE_NEW = 'http://edukg.org/knowledge/3.0/instance/%s#%s-%s'
CLASS_TEMPLATE_NEW = 'http://edukg.org/knowledge/3.0/class/%s#%s-%s'

COURSES_ZH = ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治"]
COURSES_EN = ['chinese', 'math', 'english', 'physics', 'chemistry', 'biology', 'history', 'geo', 'politics']

TO_ZH_TR = lambda x: zhconv.convert(x, 'zh-tw')
TO_ZH_SI = lambda x: zhconv.convert(x, 'zh-cn')

EDUKG_INS = Namespace('http://edukg.org/knowledge/3.0/instance/')
EDUKG_CLASS = Namespace('http://edukg.org/knowledge/3.0/class/')
EDUKG_PROP = Namespace('http://edukg.org/knowledge/3.0/property/')
EDUKG_ANNOT = Namespace('http://edukg.org/annotation#')
EDUKG_CATEGORY = Namespace('http://edukg.org/knowledge/3.0/category#')

NAMESPACE2URI = {
    'edukg_prop': EDUKG_PROP,
    'edukg_ins': EDUKG_INS,
    'edukg_cls': EDUKG_CLASS,
    'edukg_annotation': EDUKG_ANNOT,
    'edukg_category': EDUKG_CATEGORY,
    'purl_element': URIRef('http://purl.org/dc/elements/1.1/description'),
    'foaf': FOAF,
    'owl': OWL,
    **{
        'edukg_%s_%s' % (u, course): EDUKG_PROP[course + '#'] if u == 'prop' else EDUKG_CLASS[
            course + '#'] if u == 'cls' else EDUKG_INS[course + '#']
        for course in COURSES_EN + ['common', 'common_candidate'] for u in ['prop', 'ins', 'cls']
    }
}

REMOVED_KEYWORDS = [
    "电视剧",
    "电影",
    "纪录片",
    "游戏",
    "体育",
    "娱乐",
    "明星",
    "网络小说",
    "手游",
    "歌曲",
    "微博",
    "运动员",
    "节目",
    "公司",
    "微博",
    "角色",
    "动画",
    "儿歌",
]

OTHERS_CLASSES = [
    "游戏",
    "娱乐人物",
    "体育",
    "网络小说"
]

isEntity = lambda v: isinstance(v, str) and (v in IMPLICIT_CLASSES or v.startswith('http://edukb.org/') or v.startswith(
    'http://www.w3.org') or v.startswith('http://edukg.org/'))
# Is the node that starts with "kb.cs.tsinghua.edu.cn" a entity or a string value? - Confirmed! it is a value
getDomain = lambda v: re.search('[http|https]://(.+?)/(.*)', v)[1]
getSuffix = lambda v: re.search('[http|https]://(.+?)/(.*)', v)[2]
getPredicateName = lambda v: v.split('/')[-1].split('-')[0] if 'edukg' in v or 'edukb' in v else v
getCourse = lambda x: re.search('instance/(.+)#', x)[1] if re.search('instance/(.+)#', x) else None


# def read_csv_triplets(file_path):
#     df = pd.read_csv(file_path, sep='\t', names=['s', 'p', 'o'])
#     return df

def validate_processed_triplets(df):
    if all([col in df.columns for col in ['s', 'p', 'o']]):
        judgements = [
            all(df['s'].apply(lambda x: isinstance(x, str) and x.startswith('http')).tolist()),
            all(df['p'].apply(lambda x: isinstance(x, str) and x.startswith('http')).tolist()),
            all(df['o'].apply(lambda x: isinstance(x, str)).tolist()),
        ]
        return all(judgements)
    else:
        return False


def read_csv_triplets(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        content = f.read()
        lines = content.splitlines()
    try:
        processed = {
            's': [],
            'p': [],
            'o': [],
        }
        buffer = ''
        for line in lines:
            buffer += line  # Append the current line to a buffer
            c = buffer.count('\t')
            if c == 2:
                s, p, o = line.split('\t')
                processed['s'].append(s)
                processed['p'].append(p)
                processed['o'].append(o)
                buffer = ''
            elif c > 2:
                raise Exception("More than three elements exist in a row!")
        df = pd.DataFrame(processed)
        if validate_processed_triplets(df):
            return df, True
    except Exception as e:
        print(e)
        pass
    print("########## Reading raw triplets from %s ##########" % file_path)
    items = [v.split('\t') for v in content.split('\n')]
    items = [v if len(v) <= 3 else [v[0], v[1], '\t'.join(v[2:])] for v in items]
    err_indices = [i for i in range(len(items)) if len(items[i]) < 3 or not items[i][1].startswith('http')]
    err_segments = []
    for index in err_indices:
        if index - 1 not in err_indices:
            err_segments.append(index - 1)
        if index + 1 not in err_indices:
            err_segments.append(index + 1)
    segments = [0, *err_segments, len(items)]
    err_segments = [(err_segments[2 * i], err_segments[2 * i + 1]) for i in range(int(len(err_segments) / 2))]
    segments = [(segments[2 * i], segments[2 * i + 1]) for i in range(int(len(segments) / 2))]
    new_items = [v for segment in segments for v in items[segment[0]:segment[1]]]
    for err_segment in err_segments:
        new_items.append([items[err_segment[0]][0], items[err_segment[0]][1], '\n'.join(
            [items[err_segment[0]][2]] + ['\t'.join(items[i]) for i in range(err_segment[0] + 1, err_segment[1])])])
    df = pd.DataFrame({'s': [v[0] for v in new_items], 'p': [v[1] for v in new_items], 'o': [v[2] for v in new_items]})
    return df, validate_processed_triplets(df)


def analyze_triplets(df):
    print("########## Analyzing raw triplets ##########")
    result = {'entity': {}, 'relation': {}}
    # Gather nodes (including entities and valuies) and predicates (including object properties and data properties)
    print("##### Gathering nodes and predicates #####")
    assert isinstance(df, pd.core.frame.DataFrame)
    predicates = [v for v in list(set(df['p'])) if isinstance(v, str)]
    nodes = list(set(list(df['s']) + list(df['o'])))
    entities, values = [], []
    for n in nodes:
        if isEntity(n):
            entities.append(n)
        else:
            values.append(n)
    result['entity']['num'] = len(entities)
    # Analyze entities in each domain
    print("##### Analyzing entities #####")
    entity_domains = {v: getDomain(v) for v in entities if isinstance(v, str)}
    entity_domains_cnt = {v: list(entity_domains.values()).count(v) for v in set(entity_domains.values())}
    result['entity']['entity_domain_cnt'] = entity_domains_cnt

    # Analyze entities in different domains
    entity_subdomain_info = {}
    for domain in entity_domains_cnt:
        suffixes = [getSuffix(v).split('/') for v in entity_domains if
                    isinstance(v, str) and entity_domains[v] == domain]
        entity_subdomain_info[domain] = analyze_subdomain(suffixes)
    result['entity']['entity_subdomain_info'] = entity_subdomain_info

    # Analyze entity class info
    class_triplets = df[df['p'] == TYPE_PREDICATE]
    result['entity']['entity_type_cnt'] = {}
    for i in tqdm(range(class_triplets.shape[0])):
        tail = class_triplets.iloc[i]['o']
        if tail in result['entity']['entity_type_cnt']:
            result['entity']['entity_type_cnt'][tail] += 1
        else:
            result['entity']['entity_type_cnt'][tail] = 1

    # Analyze predicates in different domains
    print("##### Analyzing predicates #####")
    predicate_domains = {v: getDomain(v) for v in tqdm(predicates) if isinstance(v, str)}
    predicate_domains_cnt = {v: list(predicate_domains.values()).count(v) for v in
                             tqdm(set(predicate_domains.values()))}
    result['relation']['relation_domain_cnt'] = predicate_domains_cnt
    print("##### Differentiate object properties & data properties #####")
    # Differentiate object properties and data properties
    result['relation']['num'] = len(predicates)

    # Using DataFrame's groupby function and apply function for aggregation to replace the original time-consuming "is_object_property" function
    triplets_by_predicate = df.groupby('p')
    predicate_samples = dict(triplets_by_predicate['o'].progress_apply(
        lambda x: (
            is_entity := x.apply(lambda y: isEntity(y)),
            entity_objs := x.loc[is_entity].tolist(),
            data_objs := x.loc[~is_entity].tolist(),
            {
                'object': random.sample(entity_objs, min(len(entity_objs), 10)),
                'data': random.sample(data_objs, min(len(data_objs), 10)),
                'ratio': sum(is_entity) / len(x),
                'num': len(x),
            }
        )
    ))
    predicate_samples = {pred: predicate_samples[pred][-1] for pred in predicate_samples}

    object_properties, data_properties = [v for v in tqdm(predicate_samples) if predicate_samples[v]['ratio'] > .20], [v
                                                                                                                       for
                                                                                                                       v
                                                                                                                       in
                                                                                                                       tqdm(
                                                                                                                           predicate_samples)
                                                                                                                       if
                                                                                                                       predicate_samples[
                                                                                                                           v][
                                                                                                                           'ratio'] <= .20]
    result['relation']['relation_samples'] = {p: {} for p in predicates}
    for p in tqdm(object_properties):
        result['relation']['relation_samples'][p]['type'] = 'object_property'
    for p in tqdm(data_properties):
        result['relation']['relation_samples'][p]['type'] = 'data_property'
    for p in tqdm(predicate_samples):
        result['relation']['relation_samples'][p] = {
            **result['relation']['relation_samples'][p],
            **predicate_samples[p]
        }
    # for p in tqdm(object_properties + data_properties):
    #     predicate_triplets = df[df['p'].apply(lambda x: x == p)]
    #     predicate_triplets_object, predicate_triplets_data =  predicate_triplets[predicate_triplets['o'].apply(lambda x: isEntity(x))], predicate_triplets[predicate_triplets['o'].apply(lambda x: not isEntity(x))]
    #     result['relation']['relation_samples'][p]['object'], result['relation']['relation_samples'][p]['data'] = list(predicate_triplets_object.sample(n=min(len(predicate_triplets_object), 10))['o']), list(predicate_triplets_data.sample(n=min(len(predicate_triplets_data), 10))['o'])

    # Analyze each domain's predicates thorughout each suffix member
    print("##### Analyzing predicates' subdomain info #####")
    subdomain_info = {}
    for domain in tqdm(predicate_domains_cnt):
        suffixes = [getSuffix(v).split('/') for v in predicate_domains if
                    isinstance(v, str) and predicate_domains[v] == domain]
        subdomain_info[domain] = analyze_subdomain(suffixes)
    return result


# def is_object_property(property, df):
#     assert isinstance(property, str) and isinstance(df, pd.core.frame.DataFrame)
#     subset = df[df['p'] == property]
#     objects = list(subset['o'])
#     entities_in_objects = sum(isEntity(v) for v in objects)
#     # If the ratio of objects (URIs) is bigger than 20% in all subjects, then the relation can be considered as object property
#     if entities_in_objects > len(objects) * .20:
#         return True, entities_in_objects / len(objects) * 100
#     else:
#         return False, entities_in_objects / len(objects) * 100


def analyze_subdomain(suffixes):
    suffixes_max = max([len(v) for v in suffixes])
    suffix_name = [list(set(['/'.join(v[:i]) for v in suffixes])) for i in range(1, suffixes_max + 1)]
    suffix_num = [len(v) for v in suffix_name]
    # if the pattern is [1, 1, ... 1, len(suffixes)], it means that there's no subdomains for this domain's predicates
    if all([v == 1 for v in suffix_num[:-1]]) or len(suffix_num) <= 1:
        return None, suffixes
    else:
        start_i = min([v for v in range(len(suffix_num)) if
                       suffix_num[v] != 1])  # Find the first index in suffix_num that its element does not equal to 1
        subdomains = list(set(['/'.join(v[:start_i + 1]) for v in suffixes]))
        leaf_nodes = []
        subdomain_suffixes = {s: [] for s in subdomains}
        for v in suffixes:
            subdomain_suffixes['/'.join(v[:start_i + 1])].append(v)
        leaf_nodes = [subdomain_suffixes[s][0] for s in subdomain_suffixes if len(subdomain_suffixes[s]) == 1]
        subdomain_suffixes = {s: subdomain_suffixes[s] for s in subdomain_suffixes if len(subdomain_suffixes[s]) > 1}
        subdomain_result = {v: analyze_subdomain(subdomain_suffixes[v]) for v in subdomain_suffixes}
        return {
                   v.split('/')[-1]: {
                       "leaf_nodes": subdomain_result[v][1],
                       "num": len(subdomain_suffixes[v]),
                       "whole_suffix": v,
                       "sub": subdomain_result[v][0],
                       "type": "mid"
                   }
                   for v in subdomain_suffixes}, leaf_nodes


# Building networkx's graph with neo4j-admin import format dataframes
def build_network(nodes, relations):
    g = nx.Graph()
    # Adding nodes to the constructed graph
    for i in range(nodes.shape[0]):
        uri, name = nodes.iloc[i]['uri:ID'], nodes.iloc[i]['name']
        if not (isinstance(name, str) and len(name) and name != 'Unknown'):
            name = uri
        g.add_node(uri, name=name)
    # Adding relations to the constructed graph based on existed nodes
    for i in range(relations.shape[0]):
        subj, rel, obj = relations.iloc[i][':START_ID'], relations.iloc[i][':TYPE'], relations.iloc[i][':END_ID']
        g.add_edge(subj, obj, uri=rel, name=getPredicateName(rel))
    return g


def build_properties_dataframe(nodes, relations, info):
    assert isinstance(nodes, pd.DataFrame) and isinstance(relations, pd.DataFrame) and isinstance(info, dict)
    nodes.set_index('uri:ID', drop=False, inplace=True)
    data_properties, object_properties = [v for v in info['relation']['relation_samples'] if
                                          info['relation']['relation_samples'][v]['type'] == 'data_property'], [v for v
                                                                                                                in info[
                                                                                                                    'relation'][
                                                                                                                    'relation_samples']
                                                                                                                if info[
                                                                                                                    'relation'][
                                                                                                                    'relation_samples'][
                                                                                                                    v][
                                                                                                                    'type'] == 'object_property']
    data_properties, object_properties = pd.DataFrame({'uri': data_properties}), pd.DataFrame(
        {'uri': object_properties})
    data_properties['num'] = data_properties['uri'].apply(lambda x: info['relation']['relation_samples'][x]['num'])
    object_properties['num'] = object_properties['uri'].apply(lambda x: info['relation']['relation_samples'][x]['num'])
    data_properties['entity_num'] = data_properties['uri'].apply(lambda x: int(
        info['relation']['relation_samples'][x]['num'] * info['relation']['relation_samples'][x]['ratio']))
    object_properties['entity_num'] = object_properties['uri'].apply(lambda x: int(
        info['relation']['relation_samples'][x]['num'] * info['relation']['relation_samples'][x]['ratio']))
    data_properties.set_index('uri', drop=False, inplace=True)
    object_properties.set_index('uri', drop=False, inplace=True)

    for i in tqdm(range(object_properties.shape[0])):
        predicate = object_properties.iloc[i]['uri']
        chosen = relations[relations[':TYPE'] == predicate]
        for j in tqdm(range(chosen.shape[0])):
            value_s, value_o = chosen.iloc[j][':START_ID'], chosen.iloc[j][':END_ID']
            name_s, name_o = nodes.loc[value_s, 'name'], nodes.loc[value_o, 'name'] if isEntity(
                value_o) and 'category' not in value_o else value_o
            if 'annotation' in predicate or (
                    isinstance(name_s, str) and isinstance(name_o, str) and len(name_s) > 0 and len(
                    name_o) > 0 and name_s != 'Unknown' and name_o != 'Unknown'):
                object_properties.loc[predicate, 'sample_s'] = name_s
                object_properties.loc[predicate, 'sample_o'] = name_o
                break

    for i in tqdm(range(data_properties.shape[0])):
        predicate = data_properties.iloc[i]['uri']
        predicate_domain = getDomain(predicate)
        if isinstance(predicate, str) and ('edukg' in predicate_domain or 'edukb' in predicate_domain):
            property_name = getPredicateName(predicate)
            chosen = nodes[nodes[property_name].apply(lambda x: isinstance(x, str) and len(x) > 0)]
            for j in range(chosen.shape[0]):
                name_s, value_o = chosen.iloc[j]['name'], chosen.iloc[j][property_name]
                name_o = nodes.loc[value_o, 'name'] if isEntity(value_o) else value_o
                if isinstance(name_s, str) and isinstance(name_o, str) and len(name_s) > 0 and len(
                        name_o) > 0 and name_s != 'Unknown' and name_o != 'Unknown':
                    data_properties.loc[predicate, 'sample_s'] = name_s
                    data_properties.loc[predicate, 'sample_o'] = name_o
                    break

    return data_properties, object_properties


def access_node(root, x):
    node = root
    for i in x:
        node = node['nodes'][int(i)]
    return node


def get_classes_from_classtree(root):
    classes = {}
    q = queue.Queue()
    if not 'nodes' in root:
        return {}
    for i in range(len(root['nodes'])):
        q.put([i])
    while not q.empty():
        nodeIndices = q.get()
        node = access_node(root, nodeIndices)
        classes[node['uri']] = {'word': node['word'], 'hierarchy': '-'.join([str(i) for i in nodeIndices])}
        if 'nodes' in node:
            for j in range(len(node['nodes'])):
                q.put(nodeIndices + [j])
    return classes


def get_classtree_str(root, indent=0):
    s = ""
    if 'nodes' in root:
        s += "\t" * indent + root['word'] + "\n"
        for v in root['nodes']:
            s += get_classtree_str(v, indent=indent + 1)
        return s
    else:
        return "\t" * indent + root['word'] + "\n"


def getURI(data):
    i = 0
    while i < data.shape[0]:
        uri = data.iloc[i]['s']
        if data[(data['s'] == uri) | (data['o'] == uri)].shape[0] == data.shape[0]:
            return uri
    return None


def read_ttl(file_path):
    g = Graph()
    g.parse(file_path, format='ttl')
    triplets = {
        's': [],
        'p': [],
        'o': []
    }
    for (s, p, o) in tqdm(g):
        triplets['s'].append(str(s))
        triplets['p'].append(str(p))
        triplets['o'].append(str(o))
    return pd.DataFrame(triplets)


def parse_subclass_network(file_path):
    subclass_file = [v for v in os.listdir(file_path) if 'subclassOf' in v][0]
    subclass_ttl = read_ttl(os.path.join(file_path, subclass_file))
    concept_file = [v for v in os.listdir(file_path) if 'concept' in v][0]
    concept_ttl = read_ttl(os.path.join(file_path, concept_file))
    classnames = concept_ttl[concept_ttl['p'] == LABEL_PREDICATE]

    g = nx.DiGraph()
    g.add_nodes_from(classnames['s'].tolist())
    for i in range(classnames.shape[0]):
        g.nodes[classnames.iloc[i]['s']]['name'] = classnames.iloc[i]['o']
    g.add_edges_from([(subclass_ttl.iloc[i]['s'], subclass_ttl.iloc[i]['o']) for i in range(subclass_ttl.shape[0])])

    return g


def find_ancestors(g, uri):
    q = queue.Queue()
    result = {}
    q.put(uri)
    while not q.empty():
        u = q.get()
        node = dict(g.nodes[u]) if u in g.nodes else {}
        result[u] = g.nodes[u]['name'] if 'name' in node else None
        fathers = list(g.successors(u))
        for f in fathers:
            if not f in result:
                q.put(f)
    return result


def find_descendants(g, uri):
    q = queue.Queue()
    result = {}
    q.put(uri)
    while not q.empty():
        u = q.get()
        node = dict(g.nodes[u]) if u in g.nodes else {}
        result[u] = g.nodes[u]['name'] if 'name' in node else None
        children = list(g.predecessors(u))
        for f in children:
            if not f in result:
                q.put(f)
    return result


def get_k_fold_dataset_by_class(data, dtype='seed_df', label_key='class', n=2):
    skf = StratifiedKFold(n_splits=n)
    result = []
    if dtype == 'seed_df':
        data['instance_list'] = data['instances'].apply(
            lambda x: [v for v in x.split(',')] if isinstance(x, str) else [])
        data = data[['class', 'instance_list']].to_dict('records')
        data = {item['class'].strip(): item['instance_list'] for item in data if len(item['instance_list']) > 2}
        x, y = np.array([ins for cls in data for ins in data[cls]]), np.array(
            [cls for cls in data for ins in data[cls]])
    elif dtype == 'seed_json_list':
        x, y = np.array([v['x']['uri'] for v in data]), np.array([v['y'] for v in data])
    elif dtype == 'seed_excel':
        x, y = np.array(data['uri']), np.array(data['y'])
    skf.get_n_splits(x, y)
    for train_i, test_i in skf.split(x, y):
        x_train, x_test = x[train_i], x[test_i]
        y_train, y_test = y[train_i], y[test_i]
        result.append({
            'train': {
                'x': x_train,
                'y': y_train
            },
            'test': {
                'x': x_test,
                'y': y_test
            }
        })
    return result


def test_typing_algorithm(algo, data, dtype='seed_df'):
    kfold_data = get_k_fold_dataset_by_class(data, dtype=dtype)
    results = []
    for fold in kfold_data:
        algo.clear()
        algo.train(fold['train']['x'], fold['train']['y'])
        results.append(
            algo.test(fold['test']['x'], fold['test']['y'])
        )
    return results


def lowest_common_ancestor(g, m, n, search_limit=None):
    assert isinstance(g, nx.DiGraph) and isinstance(m, str) and isinstance(n, str)
    if m not in g.nodes or n not in g.nodes:
        return []
    if search_limit is None:
        gsize = g.size()
        search_limit = np.log(gsize)
    m_queue, n_queue = queue.Queue(), queue.Queue()
    m_dict, n_dict = {}, {}
    i = 0
    m_queue.put(m)
    n_queue.put(n)
    while True:
        m_nodes, n_nodes = [], []
        while not m_queue.empty():
            m_nodes.append(m_queue.get())
        while not n_queue.empty():
            n_nodes.append(n_queue.get())
        for p in m_nodes:
            m_dict[p] = i
        for p in n_nodes:
            n_dict[p] = i
        hit_node = list(set([v for v in list(m_dict) + list(n_dict) if v in m_dict and v in n_dict]))
        if len(hit_node) > 0:
            return sorted([(v, m_dict[v], n_dict[v]) for v in hit_node], key=lambda x: x[1] + x[2], reverse=True)
        else:
            for v in m_nodes:
                for p in list(g.successors(v)):
                    m_queue.put(p)
            for v in n_nodes:
                for p in list(g.successors(v)):
                    n_queue.put(p)
        i += 1
        if i > search_limit:
            return []


def get_LCA_pairs(g, m_list, n_list, search_limit=None):
    product_results = pd.DataFrame(
        [res for v in product(m_list, n_list) for res in lowest_common_ancestor(g, *v, search_limit=search_limit)],
        columns=['uri', 'm_length', 'n_length'])
    if product_results.shape[0] == 0:
        return []
    product_results['score'] = product_results.apply(
        lambda x: 169 / (x['m_length'] * x['n_length']) * np.log(g.nodes[x['uri']]['depth']) / len(
            list(g.predecessors(x['uri']))), axis=1)
    return sorted(product_results.groupby('uri')['score'].apply(sum).to_dict().items(), key=lambda x: x[1],
                  reverse=True)


def get_type_LCAs(g, classdict):
    names = list(classdict.keys())
    entity_pairs = [v for v in list(product(names, names)) if v[0] != v[1]]
    results = [get_LCA_pairs(g, classdict[pair[0]], classdict[pair[1]]) for pair in tqdm(entity_pairs)]
    return sorted(pd.DataFrame([i for v in results for i in v], columns=['uri', 'score']).groupby('uri')['score'].apply(
        sum).to_dict().items(), key=lambda x: x[1], reverse=True)


def lowest_common_ancestors(g, class_cnt, root_class='http://xlore.org/concept/zhc290957'):
    assert isinstance(g, nx.DiGraph) and isinstance(class_cnt, dict)
    class_cnt = {v: class_cnt[v] for v in class_cnt if v in g.nodes and nx.has_path(g, v, root_class)}
    if len(class_cnt) == 0:
        return []
    paths = {v: nx.shortest_path(g, v, root_class) for v in class_cnt}
    i = -1
    is_same = lambda x: len(x) > 1 and all([v == x[0] for v in x])
    min_length = min([len(paths[k]) for k in paths])
    while True:
        if i < -min_length or not is_same([paths[v][i] for v in paths]):
            break
        i -= 1
    if i < -2:
        paths = {v: paths[v][:i + 2] for v in paths}
    else:
        paths = {v: paths[v][:-1] for v in paths}
    score_by_path = {
        v: [
            (uri, 169 / ((g.nodes[v]['depth'] - g.nodes[uri]['depth']) ** 2 + 1) * np.log(g.nodes[uri]['depth']) / (
                        len(list(g.predecessors(uri))) + 1) * class_cnt[v])
            for uri in paths[v]]
        for v in paths
    }
    return sorted(
        pd.DataFrame([v for k in score_by_path for v in score_by_path[k]], columns=['uri', 'score']).groupby('uri')[
            'score'].apply(lambda x: np.sqrt(x.prod())).to_dict().items(), key=lambda x: x[1], reverse=True)


def cosine_matrix(matrix_a, matrix_b):
    assert isinstance(matrix_a, np.ndarray) and isinstance(matrix_b, np.ndarray)
    normalized_a, normalized_b = (matrix_a.T / (np.sqrt((matrix_a ** 2).sum(axis=1)))).T, (
                matrix_b.T / (np.sqrt((matrix_b ** 2).sum(axis=1)))).T
    return np.dot(normalized_a, normalized_b.T)


def cosine_vec2matrix(vec, matrix):
    assert isinstance(matrix, np.ndarray) and isinstance(vec, np.ndarray)
    normalized_mat = (matrix.T / (np.sqrt((matrix ** 2).sum(axis=1)))).T
    sim = np.dot(vec / np.sqrt((vec ** 2).sum()), normalized_mat.T)
    return sim


def cosine_vecbyvec(vec, ctx):
    assert isinstance(vec, np.ndarray) and isinstance(ctx, np.ndarray)
    return np.dot(vec, ctx) / (np.linalg.norm(vec)*np.linalg.norm(ctx))
