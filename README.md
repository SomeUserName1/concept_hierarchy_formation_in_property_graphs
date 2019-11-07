# Label/Property/Concept Hierarchy/Taxonomy Inference


Update 01.10
_______________________________________________________________________

Questions that Project should answer:  
- [x] Find label hierarchies  
- [x] Extract only robust sub-hierarchies  
- [x] Deal with noise  
- [x] How distant are two instances  
- [x] How can we make such an existing algorithm graph aware:  
            By considering the structure of a graph (in the clustering) regain deleted labels/make clustering 
            algorithms more robust  
    - for each node how many edges of each type: Introduce new properties with graphy traits/new column in the feature vector  
    - Neighbours, types of neighbours, type of relationships, cumulative relationship, ...  
- [ ] How much do we need to sample to get a correct hierarchy? Or how many distinct labels do we need?    


### Now:
- [x] Data: LDBC SNB generator
- [x] Algorithms: Single Linkage, TTSAS, OPTICS, HDBSCAN, Trestle, Cobweb
- [x] Features: label sets, neighbourhood  
- [ ] Experiments: What features shall be used?
- [x] Data: LDBC SNB Generator, maybe Yelp


Update 03.09.2019
_______________________________________________________________________
## Overall Structure: ## 
1. General:
    - [x] Make code modular and functional to execute different varieties of the pipeline

2. Data: Yelp and Synthetic 
    - [x] Make callable from python
    - [x] Create Tree from whats generated

3. Preprocessing: Binary Count Vectorization, SVD/PCA/ICA or tSNE/UMAP/MDS. 
    - [x] Binary count vectorization
    - [x] PCA-tSNE combination for data vis. and numeric only methods. 

4. Clustering:
    1. Pre-Clustering: (sklearn, pyclustering)
        - [x] Randomized search for hyper param tuning
        - [x] SkLearn: AffinityProp, Spectral, Optics, DBScan
        - [x] PyClustering: KMeans, KMedoid, KMedians, EM, BSAS, MBSAS, TTSAS
        - [x] Wrappers for PyClustering to use Pipeline and GridSearchCV from sklearn
    2. Hierarcical Clustering
        - [x] Single, Average, Complete, Ward, Robust Single, maybe divisive
    3. End-to-End Approaches
        - [x] HDBScan (hdbscan package)
        - [x] Conceptual Clustering (concept_formation package)

5. Evaluation:  
    - [x] First Phase: Shilouette Coef., CHI is optimized for in the GridSearch  
    - [x] Tree Edit Distance:  
        - [x] get TED running with appropriate costs via python  
        - [x] RSL -> .tree -> TED  
        - [x] Trestle -> .tree -> TED  
    - [x] Visualization:  
        - [x] Dendro  
        - [x] Pre-Clustering  
            

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

