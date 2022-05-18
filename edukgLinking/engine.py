from elasticsearch import Elasticsearch as ES
import json
from utils import json_paths


class Engine():
    def __init__(self) -> None:
        engine = ES()
        mappings = {
            'mappings': {
                'properties': {
                    'label': {
                        'type': 'text'
                    },
                    'uri': {
                        'type': 'text'
                    },
                    'alias': {
                        'type': 'text'
                    }
                }
            }
        }
        if engine.indices.exists("entity"):
            return
            engine.indices.delete("entity")
        res = engine.indices.create(index='entity', body=mappings, ignore=400)
        print(res)
        iid = 0
        for json_path in json_paths.values():
            print(json_path)
            with open(json_path, 'r') as f:
                contents = json.load(f)
                for content in contents:
                    iid += 1
                    engine.index(index='entity', id=iid, body=content)
            engine.indices.refresh(index='entity')

    def search(self, key, target, size=30):
        search_body = {
            'query': {
                'match': {
                    key: target
                }
            }
        }
        engine = ES()
        result = engine.search(index='entity', body=search_body, size=size)
        return result
    
    def fuzzSearch(self, key, target, size=30):
        search_body = {
            'query': {
                'match': {
                    key: {
                        'query': target,
                        'fuzziness': 'auto'
                    }
                }
            }
        }
        engine = ES()
        result = engine.search(index='entity', body=search_body, size=size)
        return result
