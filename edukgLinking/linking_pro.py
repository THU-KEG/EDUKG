from engine import Engine
import requests
from utils import csv_path
import json
import csv
import os
import re


es = Engine()
ner_url = 'http://10.1.1.39:8001/ka/ner'

entity_dict = {}
for subject in csv_path.keys():
    entity_list = {}
    csv_file = open(csv_path[subject], 'r', encoding='utf-8-sig')
    csv_iter = csv.reader(csv_file)
    first_row = True
    label_col = -1
    uri_col = -1
    xlore_col = -1
    for row in csv_iter:
        if first_row:
            for i in range(len(row)):
                if row[i] == 'label':
                    label_col = i
                elif row[i] == 'uri':
                    uri_col = i
                elif row[i] == 'xlore':
                    xlore_col = i
            first_row = False
        else:
            entity_list[row[label_col]] = row[uri_col]
    entity_dict[subject] = entity_list
    csv_file.close()

def EM(text):
    text = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9\n]+', '', text)
    text_list = re.split('\n|。|，|\.', text)
    entity_list = []
    for t in text_list:
        if t == '':
            continue
        res = requests.post(url=ner_url, json={'text': t})
        try:
            for d in json.loads(res.content.decode('utf-8'))['spans']:
                entity_list.append(d)
        except json.decoder.JSONDecodeError:
            print(t)
            print(res.content.decode('utf-8'))
            exit(0)
    return entity_list

def CG(entity_list, subject):
    label_map = {}
    for entity in entity_list:
        hits = es.fuzzSearch('label', entity['text'], subject, 1)['hits']['hits']
        if len(hits) > 0 and hits[0]['_score'] > 12.0:
            target = hits[0]['_source']['label']
            if target in label_map.keys():
                label_map[target]['where'].append([entity['start'], entity['end']])
            else:
                label_map[target] = {
                    'uri': entity_dict[subject][target],
                    'where': [[entity['start'], entity['end']]]
                }
    data = []
    for label in label_map.keys():
        data.append({
            'label': label,
            'uri': label_map[label]['uri'],
            'where': label_map[label]['where']
        })
    return data

def CG_Simple(entity_list, subject):
    label_map = {}
    for entity in entity_list:
        if entity['text'] in entity_dict[subject].keys():
            target = entity['text']
            if target in label_map.keys():
                label_map[target]['where'].append([entity['start'], entity['end']])
            else:
                label_map[target] = {
                    'uri': entity_dict[subject][target],
                    'where': [[entity['start'], entity['end']]]
                }
    data = []
    for label in label_map.keys():
        data.append({
            'label': label,
            'uri': label_map[label]['uri'],
            'where': label_map[label]['where']
        })
    return data

def ED(text, candidate, subject):
    return candidate
    pass

def linking(text, subject):
    return ED(text, CG_Simple(EM(text), subject), subject)

def process_json(json_path):
    json_file = open(json_path, 'r')
    content = json.load(json_file)
    json_file.close()
    subject = content['Subject']
    if subject == 'english':
        return None
    text = ''
    if content['Content'] is not None and content['Content'] != '':
        text = text + '\n' + content['Content']
    for question in content['Questions']:
        if question['Question'] is not None and question['Question'] != '':
            text = text + '\n' + question['Question']
        if question['QuestionType'] == 'choosing' and question['Choices'] is not None:
            for choice in question['Choices']:
                text = text + '\n' + choice['value']
        elif question['QuestionType'] != 'answering-essay':
            assert question['Answer'] is not None
            try:
                text = text + '\n' + question['Answer'][0]
            except Exception as e:
                print(json_path)
                print(e)
                print(question)
    
    return linking(text, subject)

if __name__ == '__main__':
    # gen_dict('dicts')
    # print(process_json('/Users/flagerlee/GaoKao_generate/json/2019GaoKao/2019_39/39_10.json'))

    json_root = '/Users/flagerlee/GaoKao_generate/json'
    path_prefix = 'temp3'
    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)
    for year_name in os.listdir(json_root):
        if year_name == '.DS_Store':
            continue
        if not os.path.exists('%s/%s' % (path_prefix, year_name)):
            os.mkdir('%s/%s' % (path_prefix, year_name))
        for exam_id in os.listdir('%s/%s' % (json_root, year_name)):
            if exam_id == '.DS_Store':
                continue
            if not os.path.exists('%s/%s/%s' % (path_prefix, year_name, exam_id)):
                os.mkdir('%s/%s/%s' % (path_prefix, year_name, exam_id))
            for problem_json in os.listdir('%s/%s/%s' % (json_root, year_name, exam_id)):
                if problem_json == '.DS_Store':
                    continue
                suffix = '%s/%s/%s' % (year_name, exam_id, problem_json)
                res = process_json('%s/%s' % (json_root, suffix))
                if res is not None:
                    with open('%s/%s/%s/%s' % (path_prefix, year_name, exam_id, problem_json.split('.')[0] + '.txt'), 'w') as f2:
                        json.dump(res, f2, ensure_ascii=False)
                else:
                    with open('%s/%s/%s/%s' % (path_prefix, year_name, exam_id, problem_json.split('.')[0] + '.txt'), 'w') as f2:
                        f2.write('[]')
                        #f2.write(str(res))