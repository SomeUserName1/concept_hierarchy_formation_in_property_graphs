# Label/Property/Concept Hierarchy/Taxonomy Inference
__Presentation: 25th July 14:00__

## Overall Structure: ## 
1. General:
    - [x] Make code modular and functional to execute different varieties of the pipeline

2. Data: Yelp and Synthetic 
    - [x] Make callable from python
    - [x] Create Tree from whats generated
    ⁻ [ ] and hand-crafted one for Yelp data set

3. Preprocessing: Binary Count Vectorization, SVD/PCA/ICA or tSNE/UMAP/MDS. 
    - [x] Binary count vectorization
    - [x] tSNE combination for data vis. and numeric only methods. 

4. Clustering:
    1. Pre-Clustering: (sklearn, pyclustering)
        - [x] Randomized search for hyper param tuning
        - [x] SkLearn: AffinityProp, Spectral, Optics, DBScan
        - [x] PyClustering: KMeans, KMedoid, KMedians, EM, BSAS, MBSAS, TTSAS
        - [x] Wrappers for PyClustering to use Pipeline and GridSearchCV from sklearn
    2. Hierarcical Clustering
        - [x] Single, Robust Single
    3. End-to-End Approaches
        - [x] HDBScan (hdbscan package)
        - [x] Conceptual Clustering (concept_formation package)

5. Evaluation:  
    - [x] First Phase: Shilouette Coef., CHI is optimized for in the GridSearch  
    - [x] Tree Edit Distance:  
        - [x] get TED running with appropriate costs via python  
        - [x] RSL -> .tree -> TED  
        - [x] Trestle -> .tree -> TED  
    - [ ] KFold Cross Validation  
       - Take 90% dataset & create hierachy  
       - take 10% dataset & validate hierarchy  
    - [x] Visualization:  
        - [x] Dendro  
        - [x] Pre-Clustering  
            
6. Questions that Project should answer:
    1. [x] Find label hierarchies
    2. [x] Extract only robust sub-hierarchies
    3. [ ] How distant are two instances
    4. [x] Deal with noise
    5. [ ] How can we make such an existing algorithm graph aware: (impl in Trestle,
            By considering the structure of a graph (in the clustering) regain deleted labels
        - for each node how many edges of each type: Introduce new properties with graphy traits
        - Neighbours, types of neighbours, type of relationships, cumulative relationship, 
    6. [ ] How much do we need to sample to get a correct hierarchy? Or how many distinct labels do we need?  
 
Project: Ignore the graph
Belivable results of how good the results are ignoring the graph structure
Thesis: reintroduce graph structure => algos more robust


## TODO
23.07:

- [ ] Table
Cost vs accuracy per algo 

Algos   | Runtime   | Accuracy|    
|--- |--- |--- |  
|  a |  123s| 12 ted |  
- [ ] Chose graphics
- [ ] Slides ready (textual)
- [x] Yelp preview

TED: how it supports our scenario, why/why not


## Presentation:
1. Introduction
    1. Motivation: Yelp, got hierarchy of some struct, dont wanna extract by hand (ex.); In Neo4J: vehicle => car, vehicle =>bus, bus, car; need to explicitly state. But concetually there is an hioerarchy, just not explitly in the data. Cant do iut by hand so => hierarchical clustering 
    2.  Probelm definition (formal, proper terminology)
2. Solution
    1. Related Work: Algorithms, Brief how they work; include examples and weioght by how well they work
3. Evaluation
    1. Setup
        1. synthetic baseline: generation, noise intro, ground truth, measures: runtime and accuracy (ted)
        2. 1 Slide on the pipeline
    2. Results: if time also for yelp, but only if propperly
        1 Base experiment
            1. Table
            2. Graphs
        2. Noise exp 1
        3. NOise exp n
    preliminary yelp experiment for transition
        
4. Conclusions and What do we want to acheive:
    - multiple hioerarchies: either relationship or disjoint
    - hierarcical info on labels (selectivity)
    => HOW: Take structural info, neibhbourhood to infer hierarcy from noisy data again
    
    doesnt need to be exact but should make sense
