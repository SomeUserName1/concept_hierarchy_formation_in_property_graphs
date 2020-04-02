# Cobweb Summary

## Intro: Conceptual Clustering  
__Input__: Set of object descriptions (observations)   
__Output__: Classification scheme over obervations)  
__Algo__:  
- Use an evaluation function to discover classes with "good" concept descriptions (unsupervised)  
- focus on effective inference for unseen object properties, thus maximize inference abilities  
- incremental concept induction to address dynamics of non-deterministic environment  

## Background: Learning as a Search  
Two dimensions: Search Control & Search direction  

**Learning by Example** (Supervised learning): __Search through the space of Concept Descriptions__  
- Assumes objects are pre-classified  
- Goal is to derive concepts that approximate a description of the class  
- Search control strategy: exhaustive vs. hill climbing  
- Direction of Search: generalization (bottom-up) vs. specialization (top down)  

**Learning by Observation** (Unsupervised Learning): __Search through space of object clusters X Concept descriptions X hierarchical structures__  
- Assumes no extra information

- Actually two problems/ Search goals:  
    1. Clusering Problem: Determining useful subsets of an object set, i.e. identifying a set of object classes beeing extentional/nested sets of objects.  
    2. Characterization Problem: Determining useful concepts for each class.  
    3. Hierarchization Problem: Determine a well-structured hierarchy.  
- Search control: Again for 1., 2., 3. Exhaustive vs. Hill-climbing  
- Search direction: as above, with divisive, agglomerative additionally for hierarchies

- Incrementally Searching by building variants of Quinlan's ID3. => Significantly reduce searching costs per object (esp. for new observations with previously unseen and unknown type).  
- Extend operator set to do both search directions at a time (i.e. generalization & specialization) to recover from bad search paths avoiding explicit bachtracking.  

## Cobweb: Incremental conceptual clustering  
1. Heuristic Evaluation Measure: Category Utility  
    - Initially developed by Gluck & Corter (1985) as means of quantifying inference-related optimality  
    - Probabilistic intra-cluster similarity, inter-cluster dissimilarity tradeoff measure
        - Intra-Cluster similarity: Probability of matching attributes in a class P(A_i = V_ij | C_k)  
        - Inter-Cluster dissimilarity: Probability of non-matching attributes between classes P(C_k | A_i = V_ij)  
        - Weighting factor P(A_i = V_ij) to increase importance of predictiveness and class-based prediatbility of frequent attribute values
        - Bayes rules applied P(A_i = V_ij)P(C_k | A_i = V_ij) = P(C_k)P(A_i = V_ij | C_k)
        - Gives: Sum_k P(C_k) Sum_i Sum_j P(A_i = V_ij | C_k)^2
    - Uses Probability matching instead of probability optimization:
    Optimization is Supperior when maximizing reward, but not when adapting to distributions, thus not suitable for heuristically ordering object partitions.  
    Gigerenzer, G., & Fiedler, K. (forthcoming). Minds in Environments: The Potential of an Ecological Approach to Cognition:  "if one considers a natural environment in which animals are not as socially isolated as in a T-maze and in which they compete with one another for food, the situation looks different. Assume that there are a large number of rats and two patches, left and right, with an 80:20 distribution of food. If all animals maximized on an individual level, then they all would end up in the left part, and the few deviating from this rule might have an advantage. Under appropriate conditions, one can show that probability matching is the more rational strategy in a socially competitive environment".  
    Same for partitioning: Maximizing probability would yield uniform objects when generating from classification scheme, thus negating any intra class variance which results in a highly biased partition based on maximal attribute probability per class.
    - Finally CU is defined as the increase in the expected number of attribute values that can be correctly guessed, given a partition over the expected number f correct guesses not knowing the class:
        Sum_k P(C_k) [Sum_i Sum_j P(A_i = V_ij | C_k)^2 - Sum_i Sum_j P(A_i = V_ij)^2 ] / n  
    - If Attribute value is independent of class membership then CU = 0, thus irrelevant for class formation.

2. State Representation of Concepts 
    - Tree of Probabilistic Concepts per node
    - Probabilistic Concept: List of Attribute values with associated counts to compute the probabilities.  
    - Unlike decision trees: Each node is labeled with the descriptor (not arcs); Classification is done by partial matching, descending the tree along the best matching nodes.  
3. Operators   
One Operator per level of descend.  
    - Classifying wrt an existing class: for each partition: calculate the CU with the node added. Add to the one with highest CU  
    - creating a new class: Compare the best existing host partition's CU to the CU of a new singleton class. Chose the one with highest CU  
    - combining two classes into a single: Take two nodes and append them as children to a newly created node with additive prob. concept. Only the two best hosts are considered for merging when the resulting CU is superior over operators 1 & 2 results
    - splitting a class into several: Take the best fitting host, remove it, promote the children and add to the best fitting child directly if CU is higher as the above
    - node promotion: take a child and move it one partition higher. 

4. Control Strategy

'''
fn cobweb(observation, tree) {
    update_root_counts(observation, tree.getRoot());
    if (tree.getRoot().isLeaf()) {
        return expand(tree, observation);
    } else {
        cu_host = find_host(tree, observation);
        cu_new = new_node(tree, node, observation);
        cu_merge = merge(tree, node, observation);
        cu_split = split(tree, node, observation); // 
        maximizeCU(cu_host, cu_new, cu_merge, cu_split); // updates tree
    }
}
'''

## Evaluation
Diettrichs learning model
Normative values: high P(C|A = V) and P(A = V | C) or one of them very high

## Conclusion
