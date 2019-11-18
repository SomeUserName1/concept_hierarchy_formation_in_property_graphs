package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import javax.annotation.concurrent.ThreadSafe;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.Transaction;

import kn.uni.dbis.neo4j.conceptual.util.MathUtils;
import kn.uni.dbis.neo4j.conceptual.util.TreeUtils;


/**
 * Application of Cobweb to the property graph model.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@ThreadSafe
public class PropertyGraphCobweb {
  /** Logger. */
  static final Logger LOG = Logger.getLogger("PropertyGraphCobweb");

  /** tree root for the node properties. */
  private final AtomicReference<ConceptNode> nodePropertiesTree = new AtomicReference<>(new ConceptNode().root());
  /** Cobweb tree for the relationship properties. */
  private final AtomicReference<ConceptNode> relationshipPropertiesTree =
      new AtomicReference<>(new ConceptNode().root());
  /** Cobweb tree for the node summary. */
  private final AtomicReference<ConceptNode> nodeSummaryTree = new AtomicReference<>(new ConceptNode().root());

  /**
   * Integrates a neo4j node into the cobweb trees.
   * 1. incorporates the node properties and labels and it's relationships types and properties into the tree
   * 2. takes the results from 1, extracts structural features, creates a new node that incorporates all information
   * and integrates it into the tree
   *
   * @param db The GraphDatabaseService to be used
   * @param nodes the list of nodes of the graph to be incorporated
   * @param relationships the relationships of the graph to be incorporated
   */
  public void integrate(final GraphDatabaseService db, final List<Node> nodes, final List<Relationship> relationships) {
    // Static categorization according to properties, labels and relationship type
      ExecutorService threadPool = Executors.newFixedThreadPool(8);
      Runnable cobwebRunnable;
      for (Node node : nodes) {
        cobwebRunnable = () -> {
          try (Transaction ignored = db.beginTx()) {
            final ConceptNode properties = new ConceptNode(node);
            Cobweb.cobweb(properties, this.getNodePropertiesTree());
          }
        };
        threadPool.execute(cobwebRunnable);
      }


      for (Relationship rel : relationships) {
        cobwebRunnable = () -> {
          try (Transaction ignored = db.beginTx()) {
            final ConceptNode properties = new ConceptNode(rel);
            Cobweb.cobweb(properties, this.getRelationshipPropertiesTree());
          }
        };
        threadPool.execute(cobwebRunnable);
      }

      threadPool.shutdown();
      try {
        threadPool.awaitTermination(1, TimeUnit.HOURS);
      } catch (final InterruptedException e) {
        Thread.currentThread().interrupt();
        e.printStackTrace();
      }

      threadPool = Executors.newWorkStealingPool();

      final Future<Integer> nodeCutoff = threadPool.submit(() -> MathUtils.log2(TreeUtils.deepestLevel(
          this.getNodePropertiesTree())));
      final Future<Integer> relCutoff =
          threadPool.submit(() -> MathUtils.log2(TreeUtils.deepestLevel(this.getRelationshipPropertiesTree())));

      threadPool.execute(() -> TreeUtils.labelTree(this.getNodePropertiesTree(), "", "n"));
      threadPool.execute(() -> TreeUtils.labelTree(this.getNodePropertiesTree(), "", "r"));

      threadPool.shutdown();

      int cutoffLevelNodes = 0;
      int cutoffLevelRelationships = 0;
      try {
        threadPool.awaitTermination(1, TimeUnit.HOURS);
        cutoffLevelRelationships = relCutoff.get();
        cutoffLevelNodes = nodeCutoff.get();
      } catch (final InterruptedException | ExecutionException e) {
        Thread.currentThread().interrupt();
        e.printStackTrace();
      }


      threadPool = Executors.newWorkStealingPool();
      for (Node node : nodes) {
        final int finalCutoffLevelNodes = cutoffLevelNodes;
        final int finalCutoffLevelRelationships = cutoffLevelRelationships;
        cobwebRunnable = () -> {
          try (Transaction ignored = db.beginTx()) {
            final ConceptNode summarizedNode = new ConceptNode();
            final List<Value> co = new ArrayList<>();

            summarizedNode.setId(Long.toString(node.getId()));
            ConceptNode properties = TreeUtils.findById(Long.toString(node.getId()), this.getNodePropertiesTree());
            assert properties != null;
            String label = properties.getCutoffLabel(finalCutoffLevelNodes);
            co.add(new NominalValue(label));
            summarizedNode.getAttributes().put("NodePropertiesConcept", co);

            co.clear();
            for (Relationship rel : node.getRelationships()) {
              properties = TreeUtils.findById(Long.toString(rel.getId()), this.getNodeSummaryTree());
              assert properties != null;
              label = properties.getCutoffLabel(finalCutoffLevelRelationships);
              final NominalValue check = new NominalValue(label);
              if (co.contains(check)) {
                co.get(co.indexOf(check)).update(check);
              } else {
                co.add(check);
              }
            }
            summarizedNode.getAttributes().put("RelationshipConcepts", co);

            extractStructuralFeatures(node, summarizedNode);
            Cobweb.cobweb(summarizedNode, this.getNodeSummaryTree());
          }
        };
        threadPool.execute(cobwebRunnable);
      }

      threadPool.shutdown();
      try {
        threadPool.awaitTermination(1, TimeUnit.HOURS);
      } catch (final InterruptedException e) {
        Thread.currentThread().interrupt();
        e.printStackTrace();
      }

      TreeUtils.labelTree(this.getNodeSummaryTree(), "", "l");
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
    return this.nodePropertiesTree.get();
  }

  /**
   * Getter for the relationship properties Cobweb tree.
   * @return the cobweb instance for relationship properties
   */
  public ConceptNode getRelationshipPropertiesTree() {
    return this.relationshipPropertiesTree.get();
  }

  /**
   * Getter for the node summaries Cobweb tree.
   * @return the cobweb instance for node summaries.
   */
  public ConceptNode getNodeSummaryTree() {
    return this.nodeSummaryTree.get();
  }
}
