package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;


/**
 * Implementation of Cobweb for the property graph model.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class PropertyGraphCobweb {
  /**
   * The Root Concept, corresponding to Any.
   */
  private final ConceptNode root;

  /**
   * standard constructor, creating a root node.
   */
  public PropertyGraphCobweb() {
    this.root = new ConceptNode("Root");
  }

  /**
   * convenience method for the one below.
   */
  private void cobweb(final ConceptNode newNode, final ConceptNode currentNode) {
    this.cobweb(newNode, currentNode, true);
  }

  /**
   * run the actual cobweb algorithm.
   *
   * @param newNode       node to incorporate
   * @param currentNode   node currently visiting
   */
  public void cobweb(final ConceptNode newNode, final ConceptNode currentNode, final boolean update) {
    if (update) {
      currentNode.updateCounts(newNode);
    }
    if (currentNode.getChildren().isEmpty()) {
      currentNode.addChild(newNode);
      newNode.setParent(currentNode);
    } else {
      final double[] results = new double[4];

      Result Result = this.findHost(currentNode, newNode);
      results[0] = Result.cu;
      final ConceptNode host = Result.node;

      results[1] = this.createNewNodeCU(currentNode, newNode);
      results[2] = this.mergeNodesCU(currentNode, host, newNode);
      results[3] = this.splitNodesCU(host, newNode);

      double best = Integer.MIN_VALUE;
      int bestIdx = -1;
      for (int i = 0; i < results.length; i++) {
        if (results[i] > best) {
          best = results[i];
          bestIdx = i;
        }
      }

      switch (bestIdx) {
        case 1:
          this.createNewNode(currentNode, newNode, true);
          break;
        case 2:
          ConceptNode mergedNode = mergeNodes(currentNode, host, newNode, true);
          if (mergedNode == null) {
            throw new RuntimeException("Unreachable");
          }
          this.cobweb(newNode, mergedNode);
          break;
        case 3:
          splitNodes(host, currentNode, true);
          this.cobweb(newNode, root, false);
          break;
        case 0:
          this.cobweb(newNode, host);
          break;
        default:
          throw new RuntimeException("Invalid best operation");
      }
    }
  }

  /**
   * finds the child most suitable to host the new node.
   *
   * @param parent  node currently visiting
   * @param newNode node to be hosted
   * @return the best host and the cu for that
   */
  private Result findHost(final ConceptNode parent, final ConceptNode newNode) {
    double curCU;
    double maxCU = Integer.MIN_VALUE;
    int i = 0;
    ConceptNode clone;
    ConceptNode best = null;
    ConceptNode parentClone;
    final double parentEAP = this.getExpectedAttributePrediction(parent);

    for (ConceptNode child : parent.getChildren()) {
      clone = new ConceptNode(child);
      clone.updateCounts(newNode);
      parentClone = new ConceptNode(parent);
      parentClone.getChildren().set(i, clone);
      curCU = this.computeCU(parentClone, parentEAP);
      if (maxCU < curCU) {
        maxCU = curCU;
        best = child;
      }
      i++;
    }
    return new Result(maxCU, best);
  }

  /**
   * creates a new child for the node to incorporate and computes the cu.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   * @return the op result including the cu and the altered node
   */
  private double createNewNodeCU(final ConceptNode currentNode, final ConceptNode newNode) {
    final ConceptNode clone = new ConceptNode(currentNode);
    createNewNode(clone, newNode, false);
    return this.computeCU(clone);
  }

  /**
   * creates a new child for the node to incorporate.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   */
  private void createNewNode(final ConceptNode currentNode, final ConceptNode newNode, final boolean setParent) {
    currentNode.addChild(newNode);
    if (setParent) {
      newNode.setParent(currentNode);
    }
  }
  
  /**
   * clones the current node, splits & computes the cu.
   *
   * @param host    node to be split
   * @param current node to append children of host
   * @return op result including altered node and the cu
   */
  private double splitNodesCU(final ConceptNode host, final ConceptNode current) {
    final ConceptNode currentClone = new ConceptNode(current);
    this.splitNodes(host, currentClone, false);
    return this.computeCU(currentClone);
  }

  /**
   * splits the host node and appends its children to the current node.
   *
   * @param host    node to be split
   * @param current node to append children of host
   */
  private void splitNodes(final ConceptNode host, final ConceptNode current, final boolean setParent) {
    for (ConceptNode child : host.getChildren()) {
      if (setParent) {
        child.setParent(current);
      }
      current.addChild(child);
    }
    current.getChildren().remove(host);
  }

  /**
   * clones the actual current node, merges and computes the cu.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @param newNode node to be incorporated
   * @return op result including altered node and the cu
   */
  private double mergeNodesCU(final ConceptNode current, final ConceptNode host, final ConceptNode newNode) {
    final ConceptNode clonedParent = new ConceptNode(current);

    return (this.mergeNodes(clonedParent, host, newNode, false) != null) ? this.computeCU(clonedParent)
        : Integer.MIN_VALUE;
  }

  /**
   * merges the host node with the next best host and appends its children to the resulting node.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @param newNode node to be incorporated
   */
  private ConceptNode mergeNodes(final ConceptNode current, final ConceptNode host, final ConceptNode newNode,
                                 final boolean setParent) {
    current.getChildren().remove(host);
    final Result secondHost = this.findHost(current, newNode);
    if (secondHost.node == null) {
      return null;
    }
    current.getChildren().remove(secondHost.node);
    final ConceptNode mNode = new ConceptNode(host);
    mNode.updateCounts(secondHost.node);
    mNode.addChild(host);
    mNode.addChild(secondHost.node);
    if (setParent) {
      host.setParent(mNode);
      secondHost.node.setParent(mNode);
    }
    current.getChildren().add(mNode);

    return mNode;
  }

  /**
   * computes the category utility as defined by fisher 1987.
   *
   * @param parent the node which acts as reference node wrt. the children/partitions of the concepts
   * @return the category utility
   */
  private double computeCU(final ConceptNode parent) {
    double cu = 0.0;
    final double parentEAP = this.getExpectedAttributePrediction(parent);
    final double parentCount = parent.getCount();

    for (ConceptNode child : parent.getChildren()) {
      cu += (double) child.getCount() / parentCount
          * (this.getExpectedAttributePrediction(child) - parentEAP);
    }
    return cu / (double) parent.getChildren().size();
  }

  /**
   * computes the category utility as defined by fisher 1987.
   *
   * @param parent    the node which acts as reference node wrt. the children/partitions of the concepts
   * @param parentEAP precomputed Expected Attribute Prediction Probability for the parent node to avoid recomputation
   * @return the category utility
   */
  private double computeCU(final ConceptNode parent, final double parentEAP) {
    double cu = 0.0;
    final double parentCount = parent.getCount();
    for (ConceptNode child : parent.getChildren()) {
      cu += (double) child.getCount() / parentCount
          * (this.getExpectedAttributePrediction(child) - parentEAP);
    }
    return cu / (double) parent.getChildren().size();
  }

  // FIXME double check

  /**
   * Computes the Expected Attribute Prediction Probability for the current concept node.
   *
   * @param category the node for which to compute the cu for.
   * @return the EAP
   */
  private double getExpectedAttributePrediction(final ConceptNode category) {
    double exp = 0;
    final double total = category.getCount();
    double interm;
    ConceptValue con;
    NumericValue num;

    for (Map.Entry<String, List<Value>> attrib : category.getAttributes().entrySet()) {

      for (Value val : attrib.getValue()) {
        interm = 0;

        if (val instanceof NominalValue) {
          interm = (double) val.getCount() / total;
          exp += interm * interm;
        } else if (val instanceof NumericValue) {
          num = (NumericValue) val;
          exp += 1.0 / (num.getStd() / num.getMean() + 1);
        } else if (val instanceof ConceptValue) {
          con = (ConceptValue) val;
          for (Value cVal : attrib.getValue()) {
            interm += con.getFactor((ConceptValue) cVal) * cVal.getCount() / total;
          }
          exp += interm * interm;
        }
      }
    }
    return exp;
  }

  // FIXME double check

  /**
   * Integrates a neo4j node into the cobweb trees.
   * 1. incorporates the node properties and labels and it's relationships types and properties into the tree
   * 2. takes the results from 1, extracts structural features, creates a new node that incorporates all information
   * and integrates it into the tree
   *
   * @param node the node to be incorporated
   */
  public void integrate(final Node node) {
    // Static categorization according to properties, labels and relationship type
    final List<ConceptNode> propertyNodes = new ArrayList<>();
    final ConceptNode nodeProperties = new ConceptNode(node);
    propertyNodes.add(nodeProperties);

    ConceptNode relProperties;
    for (Relationship rel : node.getRelationships()) {
      relProperties = new ConceptNode(rel);
      propertyNodes.add(relProperties);
    }

    for (ConceptNode cNode : propertyNodes) {
      this.cobweb(cNode, this.root);
    }

    final ConceptNode summarizedNode = new ConceptNode();
    List<Value> co = new ArrayList<>();
    co.add(new ConceptValue(propertyNodes.get(0)));
    summarizedNode.getAttributes().put("NodePropertiesConcept", co);

    co = new ArrayList<>();
    for (int i = 1; i < propertyNodes.size(); i++) {
      co.add(new ConceptValue(propertyNodes.get(i)));
    }
    summarizedNode.getAttributes().put("RelationshipConcepts", co);

    this.extractStructuralFeatures(node, summarizedNode);

    this.cobweb(summarizedNode, this.root);
  }

  /**
   * Extracts structural features from the underlying graph.
   * Desired features to be extracted:
   * [x] EgoDegree
   * [x] EgoDegreePerType
   * [x] AvgNeighbourDegree
   * [x] AvgNeighbourDegreePerType
   * <p>
   * Resulting ConceptNode structure:
   * ConceptNode
   * *Label-based*
   * NodeConcept (property-based)
   * RelationshipConcepts (property-based)
   * <p>
   * *StructureBased*
   * EgoDegree
   * EgoDeg per RelationshipType (Characteristic set with counts)
   * AvgNeighbourDegree
   * NeighbourDegree per RelationshipType
   * |OutArcsEgoNet|
   * |InArcsEgoNet|
   *
   * @param node        from which to extract the features
   * @param conceptNode the ConceptNode to store the information to
   */
  private void extractStructuralFeatures(final Node node, final ConceptNode conceptNode) {
    final int egoDegree = node.getDegree();
    final Map<RelationshipType, Integer> egoDegPerType = new HashMap<>();
    final Map<RelationshipType, Integer> neighbourDegreePerType = new HashMap<>();
    int totalNeighbourDegree = 0;
    int neighbourDegree;
    RelationshipType relType;
    int noOutArcs = node.getDegree(Direction.OUTGOING);
    int noInArcs = node.getDegree(Direction.INCOMING);

    for (Relationship rel : node.getRelationships()) {
      relType = rel.getType();
      neighbourDegree = rel.getOtherNode(node).getDegree();
      noOutArcs += rel.getOtherNode(node).getDegree(Direction.OUTGOING);
      noInArcs += rel.getOtherNode(node).getDegree(Direction.INCOMING);
      totalNeighbourDegree += neighbourDegree;

      if (!egoDegPerType.containsKey(relType)) {
        egoDegPerType.put(relType, node.getDegree(relType));
      }
      if (neighbourDegreePerType.containsKey(relType)) {
        neighbourDegreePerType.put(relType, neighbourDegreePerType.get(relType) + neighbourDegree);
      } else {
        neighbourDegreePerType.put(relType, neighbourDegree);
      }
    }

    neighbourDegreePerType.replaceAll((k, v) -> v / egoDegPerType.get(k));

    // store features into node
    List<Value> temp = new ArrayList<>();
    temp.add(new NumericValue(egoDegree));
    conceptNode.getAttributes().put("EgoDegree", temp);

    temp = new ArrayList<>();
    temp.add(new NumericValue(totalNeighbourDegree / egoDegree));
    conceptNode.getAttributes().put("AverageNeighbourDegree", temp);

    for (Map.Entry<RelationshipType, Integer> egodegpt : egoDegPerType.entrySet()) {
      temp = new ArrayList<>();
      temp.add(new NumericValue(egodegpt.getValue()));
      conceptNode.getAttributes().put(egodegpt.getKey().name() + "_Degree", temp);
    }
    for (Map.Entry<RelationshipType, Integer> neighdegpt : neighbourDegreePerType.entrySet()) {
      temp = new ArrayList<>();
      temp.add(new NumericValue(neighdegpt.getValue()));
      conceptNode.getAttributes().put(neighdegpt.getKey().name() + "_NeighbourDegree", temp);
    }

    temp = new ArrayList<>();
    temp.add(new NumericValue(noOutArcs));
    conceptNode.getAttributes().put("EgoNetOutgoingEdges", temp);

    temp = new ArrayList<>();
    temp.add(new NumericValue(noInArcs));
    conceptNode.getAttributes().put("EgoNetIncomingEdges", temp);
  }

  // FIXME double check; partial fitting
  /**
   * matches a nodes attribute name to the most fitting ones of the root.
   * Uses the category utility to determine how good it fits.
   *
   * @param toMatch the concept node to be matched
   */
  public void match(final ConceptNode toMatch) {
    final List<String> rootAttribs = new ArrayList<>(this.root.getAttributes().keySet());
    final List<String> toMatchAttrib = new ArrayList<>(toMatch.getAttributes().keySet());

    final double baseEAP = this.getExpectedAttributePrediction(toMatch);
    final double[][] costMatrix = new double[toMatchAttrib.size()][rootAttribs.size()];
    ConceptNode altered;
    int i = 0;
    int j = 0;
    double min;
    final int[] minIdx = new int[toMatchAttrib.size()];
    for (String toMatchName : toMatchAttrib) {
      min = 1;
      for (String rootName : rootAttribs) {
        altered = new ConceptNode(toMatch);
        altered.getAttributes().put(rootName, altered.getAttributes().get(toMatchName));
        altered.getAttributes().remove(toMatchName);
        costMatrix[j][i] = 1 - (this.getExpectedAttributePrediction(altered) - baseEAP);
        if (costMatrix[j][i] < min) {
          min = costMatrix[j][i];
          minIdx[j] = i;
        }
        i++;
      }
      j++;
    }
    // Transform node
    for (j = 0; j < costMatrix.length; j++) {
      if (costMatrix[j][minIdx[j]] < 1) {
        toMatch.getAttributes()
            .put(rootAttribs.get(minIdx[j]), toMatch.getAttributes().get(toMatchAttrib.get(j)));
        toMatch.getAttributes().remove(toMatchAttrib.get(j));
      }
    }
  }

  /**
   * Getter for the root of the concept hierarchy.
   *
   * @return the root of the concept hierarchy
   */
  public ConceptNode getRoot() {
    return this.root;
  }

  /**
   * convenience method for printing.
   */
  public void print() {
    System.out.println(this.root.printRec(new StringBuilder(), 0));
  }

  /**
   * Used for passing the pair cu, node of the find hosts methods back to the calling method.
   */
  private static final class Result {
    /**
     * the cu that the operation yields.
     */
    private final double cu;
    /**
     * the node with the operation applied.
     */
    private final ConceptNode node;

    /**
     * Constructor.
     *
     * @param cu        the cu that the operation yields.
     * @param node      the node with the operation applied.
     */
    private Result(final double cu, final ConceptNode node) {
      this.cu = cu;
      this.node = node;
    }
  }
}
