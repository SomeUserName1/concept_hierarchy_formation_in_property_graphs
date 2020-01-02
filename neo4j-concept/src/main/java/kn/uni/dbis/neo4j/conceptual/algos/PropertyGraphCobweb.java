package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import kn.uni.dbis.neo4j.conceptual.util.MathUtils;
import kn.uni.dbis.neo4j.conceptual.util.TreeUtils;
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
  /** structural features tree. */
  private final ConceptNode structuralFeaturesTree = new ConceptNode().root();
  /** Cobweb tree for the node summary. */
  private final ConceptNode nodeSummaryTree = new ConceptNode().root();

  /**
   * Integrates a neo4j node into the cobweb trees.
   * 1. incorporates the node properties and labels and it's relationships types and properties into the tree
   * 2. takes the results from 1, extracts structural features, creates a new node that incorporates all information
   * and integrates it into the tree
   *
   * @param nodes the list of nodes to be incorporated
   */
  public void integrate(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    Set<Relationship> rels = new HashSet<>();
    ConceptNode properties;
    ConceptNode structuralFeatures;
    for (Node node : nodes) {
      node.getRelationships().forEach(rels::add);
      properties = new ConceptNode(node);
      structuralFeatures = new ConceptNode();
      extractStructuralFeatures(node, structuralFeatures);
      Cobweb.cobweb(properties, this.nodePropertiesTree);
      Cobweb.cobweb(structuralFeatures, this.structuralFeaturesTree);
    }

    for (Relationship rel : rels) {
      properties = new ConceptNode(rel);
      Cobweb.cobweb(properties, this.relationshipPropertiesTree);
    }
    ConceptNode summarizedNode;
    List<Value> co;

    int cutoffLevelNodes = MathUtils.log2(TreeUtils.deepestLevel(this.nodePropertiesTree));
    int cutoffLevelRelationships = MathUtils.log2(TreeUtils.deepestLevel(this.relationshipPropertiesTree));
    int cutoffLevelStructuralFeatures = MathUtils.log2(TreeUtils.deepestLevel(this.structuralFeaturesTree));

    TreeUtils.labelTree(this.nodePropertiesTree, "", "n");
    TreeUtils.labelTree(this.relationshipPropertiesTree, "", "r");
    TreeUtils.labelTree(this.structuralFeaturesTree, "", "s");

    String label;
    for (Node node : nodes) {
      summarizedNode = new ConceptNode();
      summarizedNode.setId(Long.toString(node.getId()));

      co = new ArrayList<>();
      properties = TreeUtils.findById(Long.toString(node.getId()), this.nodePropertiesTree);
      assert properties != null;
      label = properties.getCutoffLabel(cutoffLevelNodes);
      co.add(new NominalValue(label));
      summarizedNode.getAttributes().put("NodePropertiesConcept", co);

      co = new ArrayList<>();
      properties = TreeUtils.findById(Long.toString(node.getId()), this.structuralFeaturesTree);
      assert properties != null;
      label = properties.getCutoffLabel(cutoffLevelStructuralFeatures);
      co.add(new NominalValue(label));
      summarizedNode.getAttributes().put("StructuralFeaturesConcept", co);

      co = new ArrayList<>();
      NominalValue check;
      for (Relationship rel : node.getRelationships()) {
        properties = TreeUtils.findById(Long.toString(rel.getId()), this.relationshipPropertiesTree);
        assert properties != null;
        label = properties.getCutoffLabel(cutoffLevelRelationships);
        check = new NominalValue(label);
        if (co.contains(check)) {
          co.get(co.indexOf(check)).update(check);
        } else {
          co.add(check);
        }
      }
      summarizedNode.getAttributes().put("RelationshipConcepts", co);

      Cobweb.cobweb(summarizedNode, this.nodeSummaryTree);
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
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
    if (egoDegree == 0) {
      temp.add(new NumericValue(0));
    } else {
      temp.add(new NumericValue(totalNeighbourDegree / egoDegree));
    }
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

    conceptNode.setId(Long.toString(node.getId()));
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
   * Getter for the relationship properties Cobweb tree.
   * @return the cobweb instance for relationship properties
   */
  public ConceptNode getStructuralFeaturesTree() {
    return this.structuralFeaturesTree;
  }

  /**
   * Getter for the node summaries Cobweb tree.
   * @return the cobweb instance for node summaries.
   */
  public ConceptNode getNodeSummaryTree() {
    return this.nodeSummaryTree;
  }
}
