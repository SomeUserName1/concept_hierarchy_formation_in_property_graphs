package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;


/**
 * Application of Cobweb to the property graph model.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class PropertyGraphCobweb {
  /** tree root for the node properties. */
  private final ConceptNode nodePropertiesTree = new ConceptNode().root();
  /** Cobweb tree for the relationship properties. */
  private final ConceptNode relationshipPropertiesTree = new ConceptNode().root();
  /** Cobweb tree for the node summary. */
  private final ConceptNode nodeSummaryTree = new ConceptNode().root();

  /**
   * convenience method for printing.
   */
  public void print() {
    System.out.println(this.nodePropertiesTree.printRec(new StringBuilder(), 0));
    System.out.println(this.relationshipPropertiesTree.printRec(new StringBuilder(), 0));
    System.out.println(this.nodeSummaryTree.printRec(new StringBuilder(), 0));
  }

  /**
   * Integrates a neo4j node into the cobweb trees.
   * 1. incorporates the node properties and labels and it's relationships types and properties into the tree
   * 2. takes the results from 1, extracts structural features, creates a new node that incorporates all information
   * and integrates it into the tree
   *
   * @param nodes the list of nodes to be incorporated
   */
  public void integrate(final List<Node> nodes, final List<Relationship> relationships) {
    // Static categorization according to properties, labels and relationship type

    ConceptNode properties;
    for (Node node : nodes) {
      properties = new ConceptNode(node);
      Cobweb.cobweb(properties, this.nodePropertiesTree);
    }

    for (Relationship rel : relationships) {
      properties = new ConceptNode(rel);
      Cobweb.cobweb(properties, this.relationshipPropertiesTree);
    }
    ConceptNode summarizedNode;
     List<Value> co;

     int deepestNodes = deepestLevel(this.nodePropertiesTree);
     int deepestRels = deepestLevel(this.relationshipPropertiesTree);
    System.out.println(deepestNodes);
    System.out.println(deepestRels);

    int cutoffLevelNodes = (int) Math.log(deepestNodes + 1);
    int cutoffLevelRelationships = (int) Math.log(deepestRels + 1);
    System.out.println("Node cutoff " + cutoffLevelNodes);
    System.out.println("Relations cutoff " + cutoffLevelRelationships);
    // FIXME those are wrong

      for (Node node : nodes) {
      summarizedNode = new ConceptNode();
      co = new ArrayList<>();

      summarizedNode.setId(Long.toString(node.getId()));
      properties = findById(Long.toString(node.getId()), this.nodePropertiesTree);
      assert properties != null;
      properties = properties.getCutoffConcept(cutoffLevelNodes);
      co.add(new ConceptValue(properties));
      summarizedNode.getAttributes().put("NodePropertiesConcept", co);

      co.clear();
      ConceptValue check;
      for (Relationship rel : node.getRelationships()) {
        properties = findById(Long.toString(rel.getId()), this.relationshipPropertiesTree);
        assert properties != null;
        properties = properties.getCutoffConcept(cutoffLevelRelationships);
        check = new ConceptValue(properties);
        if (co.contains(check)) {
          co.get(co.indexOf(check)).update(check);
        } else {
          co.add(check);
        }
      }
      summarizedNode.getAttributes().put("RelationshipConcepts", co);

      extractStructuralFeatures(node, summarizedNode);
      Cobweb.cobweb(summarizedNode, this.nodeSummaryTree);
    }

  }

  private static int deepestLevel(ConceptNode node) {
    if (node.getChildren().isEmpty()) {
      return 0;
    } else {
      int deepest = 0;
      int temp;
      for (ConceptNode child : node.getChildren()) {
        temp = deepestLevel(child);
        if (temp > deepest) {
          deepest = temp;
        }
      }
      return deepest + 1;
    }
  }

  /**
   * find and return the conceptnode for already incorporated conceptnodes, else returns null.
   * @param id the id of the relationship.
   * @param node the node we are currently inspecting in the relationshipPropertiesCobweb tree
   * @return the corresponding conceptnode or null
   */
  private static ConceptNode findById(final String id, final ConceptNode node) {
    if (node.getId() != null) {
      return node.getId().equals(id) ? node : null;
    } else {
      ConceptNode temp;
      for (ConceptNode child : node.getChildren()) {
        temp = findById(id, child);
        if (temp != null) {
          return temp;
        }
      }
      return null;
    }
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
  private static void extractStructuralFeatures(final Node node, final ConceptNode conceptNode) {
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

  /**
   * Getter for the node properties Cobweb tree.
   * @return the cobweb instance for node properties
   */
  public ConceptNode getNodePropertiesTree() {
    return this.nodePropertiesTree;
  }

  /**
   * Getter for the relationship properties Cobweb tree.
   * @return the cobweb instance for relationship properties
   */
  public ConceptNode getRelationshipPropertiesTree() {
    return this.relationshipPropertiesTree;
  }

  /**
   * Getter for the node summaries Cobweb tree.
   * @return the cobweb instance for node summaries.
   */
  public ConceptNode getNodeSummaryTree() {
    return this.nodeSummaryTree;
  }
}
