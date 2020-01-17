package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.Collections;
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
   *
   */
  public void integrateLabelPropStrcutChar(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, false);
      extractStructuralFeatures(node, conceptNode);
      extractCharacteristicSet(node, conceptNode);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabelStructChar(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, true);
      extractStructuralFeatures(node, conceptNode);
      extractCharacteristicSet(node, conceptNode);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabelPropStruct(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, false);
      extractStructuralFeatures(node, conceptNode);


      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabelPropChar(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, false);
      extractCharacteristicSet(node, conceptNode);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabelStruct(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, true);
      extractStructuralFeatures(node, conceptNode);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabelChar(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, true);
      extractCharacteristicSet(node, conceptNode);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabel(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, true);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateLabelProp(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode(node, false);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  public void integrateRelStruct(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    ConceptNode conceptNode;
    List<Node> nodeList = new ArrayList<>(nodes);
    Set<Relationship> rels = new HashSet<>();

    for (Node node : nodeList) {
      node.getRelationships().forEach(rels::add);
    }
    integrateRelationships(rels);
    int cutoffLevelRelationships = MathUtils.log2(TreeUtils.deepestLevel(this.relationshipPropertiesTree)) + 1;
    cutoffLevelRelationships = Math.min(cutoffLevelRelationships, 3);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      conceptNode = new ConceptNode();
      conceptNode.setId(Long.toString(node.getId()));

      extractStructuralFeatures(node, conceptNode);
      extractRelationshipConcepts(cutoffLevelRelationships, node, conceptNode);

      Cobweb.cobweb(conceptNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  private static void extractCharacteristicSet(final Node node, final ConceptNode conceptNode) {
    Set<Value> co = new HashSet<>();
    NominalValue check;
    for (Relationship rel : node.getRelationships()) {
      check = new NominalValue(rel.getType().name());
      co.add(check);
    }
    conceptNode.getAttributes().put("RelationshipTypes", new ArrayList<>(co));
  }

  private void extractRelationshipConcepts(int cutoffLevelRelationships, Node node, ConceptNode conceptNode) {
    List<Value> co;
    ConceptNode relNode;
    String label;
    co = new ArrayList<>();
    NominalValue check;
    for (Relationship rel : node.getRelationships()) {
      relNode = TreeUtils.findById(Long.toString(rel.getId()), this.relationshipPropertiesTree);
      assert relNode != null;
      label = relNode.getCutoffLabel(cutoffLevelRelationships);
      check = new NominalValue(label);
      if (co.contains(check)) {
        co.get(co.indexOf(check)).update(check);
      } else {
        co.add(check);
      }
    }
    conceptNode.getAttributes().put("RelationshipConcepts", co);
  }

  private void integrateRelationships(Set<Relationship> rels) {
    int progress;
    int i;
    int tenPercent;
    ConceptNode conceptNode;
    progress = 0;
    i = 0;
    tenPercent = rels.size() / 10;
    System.out.println("Number of Relationships " + rels.size());
    for (Relationship rel : rels) {
      conceptNode = new ConceptNode(rel, false);
      Cobweb.cobweb(conceptNode, this.relationshipPropertiesTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the relation features processed");
      }
    }
    TreeUtils.labelTree(this.relationshipPropertiesTree, "", "r");
  }

  /**
   * Integrates a neo4j node into the cobweb trees.
   * 1. incorporates the node properties and labels and it's relationships types and properties into the tree
   * 2. takes the results from 1, extracts structural features, creates a new node that incorporates all information
   * and integrates it into the tree
   *
   * @param nodes the list of nodes to be incorporated
   */
  public void integrateSubsequent(final Set<Node> nodes) {
    // Static categorization according to properties, labels and relationship type
    Set<Relationship> rels = new HashSet<>();
    ConceptNode properties;
    ConceptNode structuralFeatures;
    List<Node> nodeList = new ArrayList<>(nodes);
    Collections.shuffle(nodeList);

    int progress = 0;
    int i = 0;
    int tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
      node.getRelationships().forEach(rels::add);
      properties = new ConceptNode(node, false);
      structuralFeatures = new ConceptNode();
      extractStructuralFeatures(node, structuralFeatures);
      Cobweb.cobweb(properties, this.nodePropertiesTree);
      Cobweb.cobweb(structuralFeatures, this.structuralFeaturesTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node features processed");
      }
    }

    integrateRelationships(rels);

    ConceptNode summarizedNode;
    List<Value> co;

    int cutoffLevelNodes = MathUtils.log2(TreeUtils.deepestLevel(this.nodePropertiesTree)) + 1;
    int cutoffLevelRelationships = MathUtils.log2(TreeUtils.deepestLevel(this.relationshipPropertiesTree)) + 1;
    int cutoffLevelStructuralFeatures = MathUtils.log2(TreeUtils.deepestLevel(this.structuralFeaturesTree)) + 1;

    TreeUtils.labelTree(this.nodePropertiesTree, "", "n");
    TreeUtils.labelTree(this.structuralFeaturesTree, "", "s");

    String label;
    Collections.shuffle(nodeList);

    progress = 0;
    i = 0;
    tenPercent = nodeList.size() / 10;
    for (Node node : nodeList) {
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

      extractRelationshipConcepts(cutoffLevelRelationships, node, summarizedNode);

      Cobweb.cobweb(summarizedNode, this.nodeSummaryTree);
      ++i;
      if (i % tenPercent == 0) {
        ++progress;
        System.out.println(progress + "0% of the node summaries processed");
      }
    }
    TreeUtils.labelTree(this.nodeSummaryTree, "", "c");
  }

  /**
   * Extracts structural features from the underlying graph.
   * Desired features to be extracted:
   * [x] EgoDegree
   * [x] EgoDegreePerType
   * [x] AvgNeighbourDegree
   * |OutArcsEgoNet|
   * |InArcsEgoNet|
   *
   * @param node        from which to extract the features
   * @param conceptNode the ConceptNode to store the information to
   */
  private static void extractStructuralFeatures(final Node node, final ConceptNode conceptNode) {
    final int egoDegree = node.getDegree();
    int totalNeighbourDegree = 0;
    int neighbourDegree;
    int noOutArcs = node.getDegree(Direction.OUTGOING);
    int noInArcs = node.getDegree(Direction.INCOMING);

    for (Relationship rel : node.getRelationships()) {
      neighbourDegree = rel.getOtherNode(node).getDegree();
      noOutArcs += rel.getOtherNode(node).getDegree(Direction.OUTGOING);
      noInArcs += rel.getOtherNode(node).getDegree(Direction.INCOMING);
      totalNeighbourDegree += neighbourDegree;
    }


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
