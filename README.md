# Label/Property/Concept Hierarchy/Taxonomy Inference
This repository contains code implementing the following functionality:  
- [A survey](src/main/python/clustering_survey.py) on hierarchical clustering
- a synthetic dataset [generator](src/main/java/generator)  
- an implementation of the COBWEB algortihm for conceptual clustering as an 
[APOC procedure](src/main/java/kn.uni.dbis.neo4j/conceptual)
- several importing and preprocessing tools found in the [cypher](src/main/cypher) and 
[python](src/main/python/preproc_yelp_business.py) sub-directories


The key ideas are presented in the following presentations and thesis:
- [Survey Presentation](doc/project/Bachelor_Presentation.pdf)
- [Thesis](doc/thesis/Label_Hierarchy_Inference_in_Property_Graph_Databases.pdf)
- [Defense Presentation](doc/defense/Klopfer_Defense_Presentation.pdf)

### Survey
The survey uses all applicable algorithms from scikit-learn, some from pyclustering that are not in sklearn. It further 
uses hdbscan from scikit-contrib (slightly adapted, so it needs to be installed from lib, see below). Further it
uses TRESTLE algorithm, and a combination of the scikit-learn & pyclustering algorithms with tandard clustering 
techniques followed by hierarchical clustering.  

*Setup:*  
On Linux do the following (macOS should be similar)
```
cd lib/hdbscan
python setup.py install
pip install -r requirements.txt
```
Run it using python clustering_survey.py

#### Data Generator
The data generator generates a hierarchy consisting of labels to be clustered. Width and depth of the hierarchy are
 configurable. Width adds nodes per level incrementing a counter per level. Depth adds more levels and appends a 
 counter per level. Labels are "stacked" i.e. a node at the lowest level has |depth| labels while the root node has one. 
 At the lowest level a couple of identical instances are created so that they __MUST__ be in a 
 cluster without a doubt. Everything above depends on the algorithm but the goal is essentially to reconstruct the 
 ground truth 
it can be called from the command line using 
```
java -jar lib/synthetic_data_generator.jar
```
If you're building via maven, that jar is in the target folder 
`target/project-dataset-generator-jar-with-dependencies.jar`

#### COBWEB implementation
During the survey it turned out, that conceptual clustering has several desirable traits that the other algorithms dont 
have, such that it was chosen to be implemented with Neo4j as graph database for the purpose of extracting graph-based 
features and use them for clustering. This was done in order to evaluate if concept hierarchies capture correlations 
among properties and features well enough to potentially use them for cardinality estimation, that is for estimating the
selectivities of certain combinations of properties and graph-based features. For example if a node is queried with a 
relation of a certain type to another node of a certain type having a property with a certain value, one can look up in 
the concept tree which concept fits best to this query and how many values are approximated to be in this category. When
constructing the tree with sample then one gets an estimation of the share of data instances belonging to the concept 
under consideration. If no concept matches well enough one can still use the uniformity assumption. With some 
modifications to the conceptual clustering framework it should also be possible to retrieve strongly correlated 
selection criteria in order to build up a look-up table for non-uniform selection criteria, such that if it is not found
the probability that the selectivity is uniform is high.  

*Setup*
Before maven can be run to test or package the code one needs to initialize an additional dependency by hand 
(used for testing):
```
mvn initialize
mvn package
```
The longest running tests are disabled by default as a dataset with 100M edges is used and it can take up to 30 minutes.

For usage with Neo4j put the jar into <neo4j-home>/plugins:  
From The Neo4J 
[manual](https://neo4j.com/docs/java-reference/current/extending-neo4j/procedures-and-functions/introduction/)  
>Procedures and functions are written in Java and compiled into jar files. 
They are deployed to the database by dropping that jar file into the plugins directory on each standalone or
 clustered server.
  The database must be re-started on each server to pick up new procedures and functions. 
>Running `CALL dbms.procedures()` will display the full list of procedures available in a particular database instance,
> including user-defined procedures.   
>
I.e. to check if the plugin was registered correctly run `CALL dbms.procedures()` and see if 
`kn.uni.dbis.neo4j.conceptual.PropertyGraphCobwebStream` is shown.  

Example call of the procedure:
```
MATCH (n)
WITH COLLECT(n) AS nodes
CALL kn.uni.dbis.neo4j.conceptual.PropertyGraphCobwebStream(nodes)
YIELD node, partitionID
RETURN node, partitionID
```
Currently the return value is not properly accessible trough Cypher. Actually using the algorithm - or rather a 
modification or reimplementation that fits the purpose better - *within* the database kernel, that is in the cardinality 
estimator would make more sense. Due to a lack of time I haven't done this, neither is there concrete code to hook this 
up to the estimator (also because N4j is sparse on internal docs and information), nor are there results on the 
improvements this gives. Another point for future work is to improve the algorithm such that it returns the qunatities 
of interest, provide a more robust optimization criterion than the category utility especially for mixed data, implement
advances in the field e.g. [LABYRINTH](literature/conceptual_clust/labyrinth.cfb.pdf). Especially interesting would be 
the bayesian extension [AutoClass](literature/conceptual_clust/autoclass_revisited.pdf) or 
[learning Bayesian networks](literature/conceptual_clust/Heckerman1995_Article_LearningBayesianNetworksTheCom.pdf) 
to capture the conditional selectivities. A survery of different extensions is specified in 
[Unsupervised Learning of Probabilistic Concept Hierarchies](literature/conceptual_clust/survey_alternations.pdf).  

