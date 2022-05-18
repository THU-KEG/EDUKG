import os
from rdflib import Graph, URIRef, Literal


class KGHandler:
    def __init__(self, root, keys=['biology', 'chemistry', 'chinese',
                                   'geo', 'english', 'history',
                                   'math', 'physics', 'politics'], f='ttl', load_ttl=True):
        self.root = root
        self.keys = keys
        self.f = f

        if load_ttl:
            self.graphs = self._load_kgs_from_ttl_r('main')
        else:
            self.graphs = {}

    def _load_kgs_from_ttl_r(self, extra_keys='main'):

        def get_key(x: str):
            for key in self.keys:
                if key in x and extra_keys in x:
                    return key
            return None

        assert self.f == 'ttl'

        graph_dict = {}
        for path in os.listdir(self.root):
            key = get_key(path)
            if key is not None:
                print(f'loading {key}...')
                g = Graph()
                g.parse(self.root + path)
                graph_dict[key] = g

        return graph_dict

    def _get_entities_label(self, key: str, attr: dict = {
        'label': '<http://www.w3.org/2000/01/rdf-schema#label>',
        'cls': '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'}):

        ent_dict = {}
        g = self.graphs[key]

        for h, r, t in g:
            if h.n3() not in ent_dict.keys():
                ent_dict[h.n3()] = {}

            for key in attr.keys():
                if r.n3() == attr[key]:
                    ent_dict[h.n3()][key] = t.n3().strip().replace('"', '')

        return ent_dict

    def get_all_labels(self, attr: dict = {
        'label': '<http://www.w3.org/2000/01/rdf-schema#label>',
        'cls': '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'}):

        all_entities = {}
        for key in self.graphs.keys():
            all_entities[key] = self._get_entities_label(key, attr)

        return all_entities

    def get_1_hop_neighbors_by_label(self, label: str, subject: str) -> list:

        assert subject in self.graphs.keys()

        q = f"""
        SELECT ?entity
        WHERE {{ ?entity <http://www.w3.org/2000/01/rdf-schema#label> "{label}". }}
        """
        uri = ''
        res = list(self.graphs[subject].query(q))
        if res:
            uri = res[0][0]
            result = self.get_1_hop_neighbors_by_uri(uri, subject)

        return result

    def get_1_hop_neighbors_by_uri(self, uri, subject):
        assert subject in self.graphs.keys()
        result = []
        for triple in self.graphs[subject]:
            if uri in triple:
                result.append(triple)

        return result

    def get_context(self, uri, subject):
        triples = self.get_1_hop_neighbors_by_uri(uri, subject)
        context = ''

        for h, r, t in triples:
            # print(t)
            if isinstance(t, Literal) and not t.n3().startswith('"http'):
                context += t.n3().strip('"')
            elif h == uri and t.n3().startswith("<http://edukg.org/knowledge/3.0/instance/"):
                q = f"""
                SELECT DISTINCT ?label
                WHERE {{
                    {t.n3()} <http://www.w3.org/2000/01/rdf-schema#label> ?label
                }}
                """
                res = list(self.graphs[subject].query(q))
                label = res[0][0].n3()
                context += label.strip('"')

            elif t == uri and h.n3().startswith("<http://edukg.org/knowledge/3.0/instance/"):
                q = f"""
                SELECT DISTINCT ?label
                WHERE {{
                    {h.n3()} <http://www.w3.org/2000/01/rdf-schema#label> ?label
                }}
                """
                res = list(self.graphs[subject].query(q))
                label = res[0][0].n3()
                context += label.strip('"')

        if len(context) > 20:
            context = context[:20]
        return context

    def print_1_hop(self, label: str, subject: str):
        result = self.get_1_hop_neighbors_by_label(label, subject)
        for triple in result:
            print(triple)

    def get_class_entities(self, cls_index, subject, n=20):
        assert subject in self.graphs.keys()
        g = self.graphs[subject]
        q = f"""
        SELECT ?o
        WHERE 
        {{ 
        ?s  <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
         <http://edukg.org/knowledge/3.0/class/{subject}#main-C{cls_index}>.
         ?s <http://www.w3.org/2000/01/rdf-schema#label> ?o.
         }}
        LIMIT {n}"""
        res = [r[0] for r in list(g.query(q))]
        return res







