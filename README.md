# Graph Structuring Toolbox

Integrate with: Neo4J  
[NeoProfiler](https://github.com/moxious/neoprofiler)
Example with Cypher only on only one data set done by hand: [Barrasa, 2017](https://jbarrasa.com/2017/03/31/quickgraph5-learning-a-taxonomy-from-your-tagged-data/)


This project is intended to provide tools that help to extract further structure 
from graphs, especially Neo4J data sets, in order to support cardinality estimation  

## 1. Pre-Processing
Currently only data set specific loaders and an interface  
Shall be able to sample and filter data to induce certain conditions
(e.g. only cluster instances with a certain category as property)  

## 2. Clustering
Currently only multi-line properties of nodes.  
Shall at some point discover sub-labels/"types" for Nodes, later maybe also for
edges, incoming and outgoing sets and sub-graphs

Algorithms to implement:  
- Linkage-based clustering (single and complete linkage) (for nodes)  => Finished, bad performance!
- HDBSCAN: state of the art, performance comparable to K-Means, hierarchical, supports soft clustering (200000 2d points in < 60s reference implementation: [SkLearn](https://github.com/scikit-learn-contrib/hdbscan))
- Adapted version of chameleon clustering (for sub-graphs) (maybe considered later, if HDBSCAN is bad)

TODO: Metrics (e.g. InfoGain, Fisher Info/Score, Shilouette coef., lift, ...)  

## 3. Sub-Graph Mining
future work, maybe FP-Tree and Apriori-Based. See literature/GraphMining  
Maybe DIMSpan?

## 4. Visualization
Shall provide a wrapper to call algorithm specific visualizations 
(e.g. Dendrogram, ...) and put them into the profiler report
