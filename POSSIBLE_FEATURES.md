# List of possible graph-based Features
### and how well they worked

## General Remarks
| Observation |Object Orientation | Property Graph | Triplet Models |
|--- |--- |--- |--- 
| Focus of Representation | Instance-focused | Depending on Scope; Analysis focused | Linkage focused |
| Information encoding | set of key-value pairs belonging to an Object per row | Mixed key-value and subject-predicate-object semantics | set of relations/triplets with Subject-Predicate-Object semantics |
|  Feature Entropy | attributes are sufficient | Mixture neccessary | Graph-based features are sufficient |
| Conversion | make key-value pairs predicate-Object relations linked to subject | Either OO or Triplet approach to "normalize" | Store as key-value pairs with subject as Storage Object |
| Corresponding Normal Form | NF^2 | [NF^2, 6 NF (dependency preserving)] | 6 NF (dependency preserving) |

dependency preserving := no extra rows'd be inserted if they're not present before when transforming from 4 NF to 5 NF


## Manual aproach:
- Handcrafted yields good results
- but not in the general case
- general rules e.g. characteristic set work but may omit structures as "neighbour with relation type X has always attribute/value Y"
        - Consider Business A -[in Category]-> category C. Char set A{in_category}. Yields no information of value

=> Build automatic extraction into Trestle algorithm

### Ego Network-based
- degree of the ego; In case of directed additionally in/out degree; In case of weighted, weighted versions of all
- Number of arcs per type
- Characteristic set aka the set of all relation types of the ego
- density per type: no arcs/no possible arcs
- Homophily: Same node type/degree
- Reach: degree/total

### Recursive
#### on Existing features
#### on neighbourhood

### [Automated Feature Extraction Paper](www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf)
Assumes no labels => Only structural features
- Local: 
    - degree of the ego
    - In case of directed additionally in/out degree
    - In case of weighted, weighted versions of all
- Ego-Net:
    - number of arcs within ego net
    - incoming and outgoing set of egonet as a whole or number of arcs to 2-Neighbourhood
    - directed and weighted versions of the above
- Recursive Features := Aggregate computed over feature value among node's neighbours
    - Used aggregates: Mean, Sum, Min, Max, Variance for numeric; mode, distribution for categorical
    - Mean/Sum/... degree of all neighbours of a node
    - Mean/Sum/... number of nodes to external neighbours
    - directed and weighted version
    - Mode attribute value with support > 50%
    - Aggregations of the above

[Role Feature](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46591.pdf)

[Relational Aggregation Fn; Trestle relevant](pages.stern.nyu.edu%2F~fprovost%2FPapers%2Fclaudia-kdd03-final.pdf)

