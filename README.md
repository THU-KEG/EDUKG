# EDUKG
EDUKG: a Heterogeneous Sustainable K-12 Educational Knowledge Graph

EduKG is proposed and maintained by the Knowledge Engineering Group of Tsinghua Univerisity. It is a heterogeneous, sustainable K-12 educational knowledge graph with an interdisciplinary and fine-grained ontology. This repository consists of **38** well-constructed knowledge graphs (more than **252 million** entities and **3.86 billion** triplets) under the ontology. 

 In general, our contributions are summarized as follows:

1. An interdisciplinary, fine-grained ontology uniformly represents K-12 educational knowledge, resources, and heterogeneous data with 635 classes, 445 object properties, and 1314 datatype properties; 

2. A large-scale, heterogeneous K-12 educational KG with more than 252 million entities and 3.86 billion triplets based on the data from massive educa- tional and external resources;

3. A flexible and sustainable construction and maintenance mechanism empowers EDUKG to evolve dynamically, where we design guiding schema of the construction methodology as hot-swappable, and we simultaneously monitor 32 different data sources for incrementally infusing heterogeneous data.

## Updates
May.13rd: 
- The first version of our repository is officially online!!!
- New heterogeneous database parsed and uploaded: NBSC, BioGRID

## Repository Framework

The framework of our work is illustrated as below:

![EDUKG Overall Architecture New](https://user-images.githubusercontent.com/96781042/168043447-3a574ed0-71ed-4686-add6-cf9d207b47a3.png)

This repository contains all the resources in the EduKG as well as the toolkits developed by which the resources are collected. This ongoing project will be maintained with special care to ensure that the users can obtain fresh and timely resources for intelligent education.

## Resources

The resources of EduKG mainly consists of three parts: knowledge topics, educational resources and the external heterogeneous data. All of them are inter-connected by the fine-grained, unified ontology.

The knowledge topics resources
| Knowledge Topic Resrouces | Description                                  | Download | Size | Last Update |
|---------------------------|----------------------------------------------|----------|------|-------------|
| Main Concepts             | The main concepts involved in K-12 education | [main.ttl](https://drive.google.com/file/d/1YoPITzjk2oKoX0k_XbLplu8IUn16bg4-/view?usp=sharing) | 16.8M  |May.18th,2022|


<br>


The educational resources
| Educational Resources     | Description                                  | Download | Size | Last Update |
|---------------------------|----------------------------------------------|----------|------|-------------|
| Exercises                 |The exercises and exams of Chinese K-12 education since 2017| exercise.ttl | N/A  |May.5th,2022|
| Reading Material          |The supplimentary and mandatory reading materials for Chinese K-12 education| [material.ttl](https://drive.google.com/file/d/1xgUWMkIl5g3h_CeneCPDOR3MUlWyX06q/view?usp=sharing)   | N/A  |May.5th,2022|

**Note:** Due to copyright issues, the textual content of the reading material will not be published 


<br>


The external heterogenous resources
| External Heterogenous data| Description                                  | Download | Size | Last Update |
|---------------------------|----------------------------------------------|----------|------|-------------|
| BioGRID| An Online Interaction Respository With Data Compiled Through Comprehensive Curation Efforts.| [biogrid.zip](https://drive.google.com/file/d/1ed0ec9WdEDuCIdrd7rJOwkyoWtr2Bc5c/view?usp=sharing)   | 52.8M  |May.5th,2022|
| UniProtKB|The central hub for the collection of functional information on proteins| [uniprot.zip](https://drive.google.com/file/d/19GqxPKCupUwIOH1OWLRSZ7LdLhKIJ4dQ/view?usp=sharing)   | 410.8M  |May.5th,2022|
| New York Times|An American daily newspaper based in New York City with a worldwide readership| [nytimes.zip](https://drive.google.com/file/d/1ZV21wGPx8oE9XKwijlAIRFlNM6cCpH9A/view?usp=sharing)   | 544.6M  |May.5th,2022|
|NBSC|Public statistical data provided by National Bureau of Statistics of China| [NBS.zip](https://drive.google.com/file/d/1ItBqExrTXonsyk8EzU9b7lj0HZK536AR/view?usp=sharing)  | 8.9M  |May.5th,2022|
|HowNet|An online common-sense knowledge base unveiling inter-conceptual relations and inter-attribute relations of concepts| [HowNet.zip](https://drive.google.com/file/d/1kZg3ose06wLIYNBJ1XfXmY7GdToR2On1/view?usp=sharing)   | 8.2M  |May.5th,2022|
|WordNet|A large lexical database of English| Xingyu   | N/A  |May.19th,2022|
|Framester|A frame-based ontological resource acting as a hub between linguistic resources | [framester.zip](https://drive.google.com/file/d/1hWSGOAYpgTk5-hrCoijhLI9jPhpPVJ5T/view?usp=sharing)   | 969.7M  |May.5th,2022|
|GeoNames|The GeoNames geographical database covers all countries and contains over eleven million placenames| [geoNames.zip](https://drive.google.com/file/d/1Yve8deeTpQsqjpT2TpTmrktB1vnlECwY/view?usp=sharing)   | 459.8M  |May.5th,2022|


[| FAOSTAT|FAOSTAT provides free access to food and agriculture data for over 245 countries and territories | Xingyu   | N/A  |May.19th,2022|
|IHS|N/A| Xingyu   | N/A  |TBD|
|Thesaurus-Merriam-Webster|Synonyms, antonyms, definitions, and example sentences developed from Merriam-Webster dictionary| Xingyu   | N/A  |TBD|
|PubChem|The world's largest collection of freely accessible chemical information| Xingyu   | N/A  |May.19th,2022|]: #


[|DataBank|DataBank gives enterprises, technology, and content providers the confidence of knowing their applications and data are always secure| Xingyu   | N/A  |TBD|
|CSKG|A multi-sources knowledge graph for common sense| Xingyu   | N/A  |May.19th,2022|]: #

[|国家统计年鉴|N/A| Xingyu   | TBD  |May.19th,2022|
|第五六次人口普查数据|N/A| Xingyu   | TBD  |May.19th,2022|
|公开政府文件|N/A| Xingyu   | TBD  |May.19th,2022|
|Wikinews (Chinese)|N/A| Jiuding  | TBD  |May.19th,2022|
|Wikinews (English)|N/A| Jiuding  | TBD  |May.19th,2022|
|Wikisources (Chinese)|N/A| Jiuding  | TBD  |May.19th,2022|
|Wikisources (English)|N/A| Jiuding  | TBD  |May.19th,2022|
|Wikidictionary (Chinese)|N/A| Jiuding  | TBD  |May.19th,2022|
|Wikidictionary (English)|N/A| Jiuding  | TBD  |May.19th,2022|
|Wikivoyage (English)|N/A| Jiuding  | TBD  |May.19th,2022|]: #

## Toolkits

The toolkits for construction the EduKG is provided below:
| Name     | Description                                  |Jump to|
|---------------------------|----------------------------------------------|----------|
| Knowledge Candidate Extraction | TBD                                          |To be uploaded|
| Concept Expansion        | TBD                                          |To be uploaded| 
| Entity Linking        | TBD                                          |To be uploaded| 
| Entity Alignment        |Aligned entity with Wikidata and XLORE entity by neighborhood information| edukgea | 
| XML2TTL Parser       |A modulized tool for parsing XML into knowledge graph with ontology| xml2ttl   | 
| Rhetorical Role Typing       |Rhetorical role typing based on dependency tree | [rhetyper](https://github.com/THU-KEG/EDUKG/tree/main/rhetyper)   | 

## Reference


