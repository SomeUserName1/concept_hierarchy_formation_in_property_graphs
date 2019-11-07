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
    this.root = new ConceptNode();
    this.root.setLabel("Root");
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
    Map<Value, Integer> temp = new HashMap<>();
    temp.put(new NumericValue(egoDegree), 1);
    conceptNode.getAttributes().put("EgoDegree", temp);

    temp = new HashMap<>();
    temp.put(new NumericValue(totalNeighbourDegree / egoDegree), 1);
    conceptNode.getAttributes().put("AverageNeighbourDegree", temp);

    for (Map.Entry<RelationshipType, Integer> egodegpt : egoDegPerType.entrySet()) {
      temp = new HashMap<>();
      temp.put(new NumericValue(egodegpt.getValue()), 1);
      conceptNode.getAttributes().put(egodegpt.getKey().name() + "_Degree", temp);
    }
    for (Map.Entry<RelationshipType, Integer> neighdegpt : neighbourDegreePerType.entrySet()) {
      temp = new HashMap<>();
      temp.put(new NumericValue(neighdegpt.getValue()), 1);
      conceptNode.getAttributes().put(neighdegpt.getKey().name() + "_NeighbourDegree", temp);
    }

    temp = new HashMap<>();
    temp.put(new NumericValue(noOutArcs), 1);
    conceptNode.getAttributes().put("EgoNetOutgoingEdges", temp);

    temp = new HashMap<>();
    temp.put(new NumericValue(noInArcs), 1);
    conceptNode.getAttributes().put("EgoNetIncomingEdges", temp);
  }

  /**
   * run the actual cobweb algorithm.
   *
   * @param newNode       node to incorporate
   * @param currentNode   node currently visiting
   * @param updateCurrent weather to update the attributes and counts
   */
  private void cobweb(final ConceptNode newNode, final ConceptNode currentNode, final boolean updateCurrent) {
    if (updateCurrent) {
      currentNode.updateCounts(newNode, false);
    }
    if (currentNode.getChildren().isEmpty()) {
      currentNode.addChild(newNode);
    } else {
      final OpResult[] results = new OpResult[4];

      results[0] = this.findHost(currentNode, newNode);
      final ConceptNode host = results[0].node;

      results[1] = this.createNewNode(currentNode, newNode);
      results[2] = this.mergeNodes(currentNode, host, newNode);
      results[3] = this.splitNodes(host, newNode);

      OpResult best = results[0];
      for (OpResult result : results) {
        if (result.cu > best.cu) {
          best = result;
        }
      }

      switch (best.operation) {
        case CREATE:
          newNode.setParent(currentNode);
          currentNode.addChild(newNode);
          break;
        case SPLIT:
          for (ConceptNode child : host.getChildren()) {
            currentNode.addChild(child);
          }
          currentNode.getChildren().remove(host);
          this.cobweb(newNode, currentNode, false);
          break;
        case MERGE:
          currentNode.getChildren().remove(host);
          currentNode.getChildren().add(results[2].node);
          this.cobweb(newNode, results[2].node, true);
          break;
        case RECURSE:
          this.cobweb(newNode, host, true);
          break;
        default:
          throw new RuntimeException("Invalid best operation");
      }
    }
  }

  // FIXME double check; partial fitting

  /**
   * finds the child most suitable to host the new node.
   *
   * @param parent  node currently visiting
   * @param newNode node to be hosted
   * @return the best host and the cu for that
   */
  private OpResult findHost(final ConceptNode parent, final ConceptNode newNode) {
    double curCU;
    double maxCU = -1;
    int i = 0;
    ConceptNode clone;
    ConceptNode best = parent;
    ConceptNode parentClone;
    final double parentEAP = this.getExpectedAttributePrediction(parent);

    for (ConceptNode child : parent.getChildren()) {
      clone = new ConceptNode(child);
      clone.updateCounts(newNode, false);
      parentClone = new ConceptNode(parent);
      parentClone.getChildren().set(i, clone);
      curCU = this.computeCU(parentClone, parentEAP);
      if (maxCU < curCU) {
        maxCU = curCU;
        best = child;
      }
      i++;
    }
    return new OpResult(Op.RECURSE, maxCU, best);
  }

  /**
   * creates a new child for the node to incorporate and computes the cu.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   * @return the op result including the cu and the altered node
   */
  private OpResult createNewNode(final ConceptNode currentNode, final ConceptNode newNode) {
    final ConceptNode clone = new ConceptNode(currentNode);
    clone.addChild(newNode);
    return new OpResult(Op.CREATE, this.computeCU(clone), clone);
  }

  /**
   * splits the host node and appends its children to the current node & computes the cu.
   *
   * @param host    node to be split
   * @param current node to append children of host
   * @return op result including altered node and the cu
   */
  private OpResult splitNodes(final ConceptNode host, final ConceptNode current) {
    final ConceptNode currentClone = new ConceptNode(current);
    for (ConceptNode child : host.getChildren()) {
      currentClone.addChild(child);
    }
    currentClone.getChildren().remove(host);
    return new OpResult(Op.SPLIT, this.computeCU(current), currentClone);
  }

  /**
   * merges the host node with the next best host and appends its children to the resulting node & computes the cu.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @param newNode node to be incorporated
   * @return op result including altered node and the cu
   */
  private OpResult mergeNodes(final ConceptNode current, final ConceptNode host, final ConceptNode newNode) {
    final ConceptNode clonedParent = new ConceptNode(current);
    clonedParent.getChildren().remove(host);
    final OpResult secondHost = this.findHost(clonedParent, newNode);
    final ConceptNode mNode = new ConceptNode(host);
    mNode.updateCounts(secondHost.node, true);
    mNode.addChild(host);
    mNode.addChild(secondHost.node);
    clonedParent.getChildren().add(mNode);
    return new OpResult(Op.MERGE, this.computeCU(clonedParent), mNode);
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

    for (Map.Entry<String, Map<Value, Integer>> attrib : category.getAttributes().entrySet()) {
      for (Map.Entry<Value, Integer> val : attrib.getValue().entrySet()) {
        interm = 0;
        if (val.getKey() instanceof NominalValue) {
          interm = (double) val.getValue() / total;
          exp += interm * interm;
        } else if (val.getKey() instanceof NumericValue) {
          exp += 1.0 / ((NumericValue) val.getKey()).getStd();
        } else if (val.getKey() instanceof ConceptValue) {
          con = (ConceptValue) val.getKey();
          for (Map.Entry<Value, Integer> cVal : attrib.getValue().entrySet()) {
            interm += con.getFactor((ConceptValue) cVal.getKey()) * cVal.getValue() / total;
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
      this.cobweb(cNode, this.root, true);
    }

    final ConceptNode summarizedNode = new ConceptNode();
    Map<Value, Integer> co = new HashMap<>();
    co.put(new ConceptValue(propertyNodes.get(0)), 1);
    summarizedNode.getAttributes().put("NodePropertiesConcept", co);

    co = new HashMap<>();
    for (int i = 1; i < propertyNodes.size(); i++) {
      co.put(new ConceptValue(propertyNodes.get(i)), 1);
    }
    summarizedNode.getAttributes().put("RelationshipConcepts", co);

    this.extractStructuralFeatures(node, summarizedNode);

    this.cobweb(summarizedNode, this.root, true);
  }

  // FIXME double check

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
   * Enumerates the possible operators of Cobweb.
   */
  enum Op {
    /**
     * Create a new child under the current node.
     */
    CREATE,
    /**
     * split the current node, appending the childs to it's parent.
     */
    SPLIT,
    /**
     * merge the two best fitting hosts.
     */
    MERGE,
    /**
     * continue traversing the tree at the next lower level.
     */
    RECURSE
  }

  /**
   * Wrapper class for a Result containing the Operation that was evaluated, the cu that it yields and the node with
   * the applied change.
   */
  private final class OpResult {
    /**
     * The operation performed.
     */
    private final Op operation;
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
     * @param operation The operation performed.
     * @param cu        the cu that the operation yields.
     * @param node      the node with the operation applied.
     */
    private OpResult(final Op operation, final double cu, final ConceptNode node) {
      this.operation = operation;
      this.cu = cu;
      this.node = node;
    }
  }
}
