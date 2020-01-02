package kn.uni.dbis.neo4j.conceptual;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;

import kn.uni.dbis.neo4j.conceptual.algos.Cobweb;
import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.ConceptValue;
import kn.uni.dbis.neo4j.conceptual.algos.NominalValue;
import kn.uni.dbis.neo4j.conceptual.algos.NumericValue;
import kn.uni.dbis.neo4j.conceptual.algos.PropertyGraphCobweb;
import kn.uni.dbis.neo4j.conceptual.algos.Value;
import kn.uni.dbis.neo4j.conceptual.proc.PropertyGraphCobwebProc;
import kn.uni.dbis.neo4j.conceptual.util.TreeUtils;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBConfig;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBSetup;
import kn.uni.dbis.neo4j.eval.annotations.GraphSource;
import kn.uni.dbis.neo4j.eval.annotations.Preprocessing;
import kn.uni.dbis.neo4j.eval.annotations.Procedures;
import kn.uni.dbis.neo4j.eval.datasets.Dataset;

/**
 * Tests for the PropertyGraphCobweb algorithm.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@ExtendWith(GraphDBSetup.class)
@GraphDBConfig()
@GraphSource()
@Procedures(procedures = PropertyGraphCobweb.class)
class PropertyGraphCobwebProcTest {

  /**
   * Tests if the procedure throws runtime errors and if the resulting tree is returned.
   *
   * @param db database to execute the procedure call against
   * @param dataset the dataset specified in the class scope, used to access it's node and relationship count
   */
  @Disabled
  @Preprocessing(preprocessing = "MATCH (n) REMOVE n.name RETURN n")
  @Test
  void testCobwebSmall(final GraphDatabaseService db, final Dataset dataset) {
    try (Transaction ignored = db.beginTx()) {
      final Stream<Node> nodes = db.getAllNodes().stream();
      System.out.println(nodes.count());
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(2000))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printCutoffTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), dataset.getNodes());
      Assertions.assertEquals(this.leafCount(subtrees[1]), dataset.getArcs());
      Assertions.assertEquals(this.leafCount(subtrees[2]), dataset.getNodes());
      Assertions.assertEquals(this.leafCount(subtrees[3]), dataset.getNodes());
    }
  }

  /**
   * Tests if the procedure throws runtime errors and if the resulting tree is returned.
   *
   * @param db database to execute the procedure call against
   */
  @Disabled
  @Test
  @Preprocessing(preprocessing = "MATCH (n) REMOVE n.nodeId RETURN n")
  @GraphSource(getDataset = Dataset.Rome99)
  void testCobwebMedium(final GraphDatabaseService db) {
    try (Transaction ignored = db.beginTx()) {
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(250))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printFullTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
        System.out.println("Tree " + root.getLabel() + " clear");
      }
      // Assertions.assertEquals(this.leafCount(subtrees[0]), 2000);
      // Assertions.assertEquals(this.leafCount(subtrees[2]), 2000);
      // Assertions.assertEquals(this.leafCount(subtrees[3]), 2000);
    }
  }


  /**
   * Tests if the procedure throws runtime errors and if the resulting tree is returned.
   *
   * @param db database to execute the procedure call against
   */
  @Test
  //@Preprocessing(preprocessing = "MATCH (n) REMOVE n.nodeId RETURN n")
  @GraphSource(getDataset = Dataset.LDBC_SNB)
  void testCobwebLDBC(final GraphDatabaseService db) {
    try (Transaction ignored = db.beginTx()) {
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(2000))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printCutoffTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[2]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[3]), 2000);
    }
  }

  /**
   * Tests if the procedure throws runtime errors and if the resulting tree is returned.
   *
   * @param db database to execute the procedure call against
   */
  @Test
  @Preprocessing(preprocessing = "MATCH (n) REMOVE n.id RETURN n")
  @GraphSource(getDataset = Dataset.YELP_OO)
  void testCobwebYelpOO(final GraphDatabaseService db) {
    try (Transaction ignored = db.beginTx()) {
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(2000))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printCutoffTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[1]), 0);
      Assertions.assertEquals(this.leafCount(subtrees[2]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[3]), 2000);
    }
  }

  /**
   * Tests if the procedure throws runtime errors and if the resulting tree is returned.
   *
   * @param db database to execute the procedure call against
   */
  @Test
  @Preprocessing(preprocessing = "MATCH (n) REMOVE n.id RETURN n")
  @GraphSource(getDataset = Dataset.YELP_GRAPH)
  void testCobwebYelpGraph(final GraphDatabaseService db) {
    try (Transaction ignored = db.beginTx()) {
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(2000))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printCutoffTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[2]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[3]), 2000);
    }
  }

  /**
   * stupid.
   * @param db stupid
   */
  //@Preprocessing(preprocessing = "MATCH (n) REMOVE n.nodeId RETURN n")
  @GraphSource(getDataset = Dataset.RoadNetNY)
  @Test
  void testCobwebMediumLarge(final GraphDatabaseService db) {
    try (Transaction ignored = db.beginTx()) {
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(2000))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printCutoffTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[2]), 2000);
      Assertions.assertEquals(this.leafCount(subtrees[3]), 2000);
    }
  }

  /**
   * stupid.
   * @param db stupid
   * @param dataset stupid
   */
  //@Preprocessing(preprocessing = "MATCH (n) REMOVE n.nodeId RETURN n")
  @GraphSource(getDataset = Dataset.InternetTopology)
  @Test
  void testCobwebLarge(final GraphDatabaseService db, final Dataset dataset) {
    try (Transaction ignored = db.beginTx()) {

      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream().limit(1000))
          .findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getStructuralFeaturesTree(), tree.getNodeSummaryTree()};

      TreeUtils.printCutoffTrees(subtrees);

      for (ConceptNode root : subtrees) {
        this.checkIds(root);
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), 1000);
      Assertions.assertEquals(this.leafCount(subtrees[2]), 1000);
      Assertions.assertEquals(this.leafCount(subtrees[3]), 1000);
    }
  }

  void doCheckIds(final ConceptNode node, List<String> ids, List<String> duplicates) {
    if (node.getId() != null) {
      if (ids.contains(node.getId())) {
        duplicates.add(node.getId());
      }
      ids.add(node.getId());
    } else {
      for (ConceptNode child : node.getChildren()) {
        doCheckIds(child, ids, duplicates);
      }
    }
  }

  void checkIds(final ConceptNode node) {
    List<String> ids = new ArrayList<>();
    List<String> duplicates = new ArrayList<>();
    doCheckIds(node, ids, duplicates);
    Assertions.assertEquals(node.getCount(), ids.size());
    Assertions.assertTrue(duplicates.isEmpty(), "Duplicate ids: " + duplicates.toString());
  }

  void heckIds(final ConceptNode node, List<String> ids) {
    if (node.getId() != null) {
      Assertions.assertFalse(ids.contains(node.getId()), "Double id: " + node.getId());
      ids.add(node.getId());
    } else {
      for (ConceptNode child : node.getChildren()) {
        heckIds(child, ids);
      }
    }
  }

  /**
   * Checks that no inner node has an ID and all leafs have one.
   * @param node currently visited node
   */
  void checkLeafType(final ConceptNode node) {
    if (node.getChildren().isEmpty() && !(node.getParent() == node)) {
      Assertions.assertNotNull(node.getId());
    } else {
      Assertions.assertNull(node.getId());
      for (ConceptNode child : node.getChildren()) {
        this.checkLeafType(child);
      }
    }
  }

  /**
   * Checks that all nodes have a parent.
   * @param node currently visited node
   */
  void checkParent(final ConceptNode node) {
    if (!(node.getParent() == node)) {
      Assertions.assertNotNull(node.getParent());
    }
    for (ConceptNode child : node.getChildren()) {
      this.checkParent(child);
    }
  }

  /**
   * Checks that the count of the parent equals the sum of the children counts.
   * @param node node currently visited.
   */
  void checkPartitionCounts(final ConceptNode node) {
    int childCounts = 0;
    for (ConceptNode child : node.getChildren()) {
      childCounts += child.getCount();
    }
    Assertions.assertEquals(node.getCount(), childCounts);
    for (ConceptNode child : node.getChildren()) {
      if (child.getId() == null) {
        this.checkPartitionCounts(child);
      }
    }
  }

  /**
   * Counts the number of leafs in a tree.
   * @param node currently visited node
   * @return the number of leafs in a tree
   */
  int leafCount(final ConceptNode node) {
    if (node.getId() != null) {
      return 1;
    } else {
      int count = 0;
      for (ConceptNode child : node.getChildren()) {
        count += this.leafCount(child);
      }
      return count;
    }
  }

  /**
   * test the counts after updating values.
   */
  @Test
  @Disabled
  void testUpdateCounts() {
    final ArrayList<Value> nomList = new ArrayList<>();
    final ArrayList<Value> numList = new ArrayList<>();
    final ArrayList<Value> conList = new ArrayList<>();
    
    final NominalValue nom = new NominalValue("a");
    final NumericValue num = new NumericValue(1);
    final ConceptValue cv = new ConceptValue(new ConceptNode());

    nomList.add(nom);
    numList.add(num);
    conList.add(cv);

    final ConceptNode c1 = new ConceptNode();
    c1.getAttributes().put("A", nomList);
    c1.getAttributes().put("B", numList);
    c1.getAttributes().put("C", conList);
    final ConceptNode c2 = new ConceptNode();
    c2.getAttributes().put("C", conList);
    c2.getAttributes().put("B", numList);
    c2.getAttributes().put("A", nomList);

    c1.updateCounts(c2);

    List<Value> vals =  c1.getAttributes().get("A");
    Assertions.assertEquals(2, vals.get(vals.indexOf(new NominalValue("a"))).getCount());
    vals =  c1.getAttributes().get("B");
    Assertions.assertEquals(2, vals.get(0).getCount());
    vals =  c1.getAttributes().get("C");
    Assertions.assertEquals(2, vals.get(vals.indexOf(cv)).getCount());
  }

  /**
   * Test if nodes are created correctly.
   */
  @Disabled
  @Test
  void testCreate() {
    final ConceptNode root = new ConceptNode().root();
    final ConceptNode conceptNode = new ConceptNode();
    final NominalValue v = new NominalValue("test");
    final List<Value> val = new ArrayList<>();
    val.add(v);
    conceptNode.getAttributes().put("name", val);
    final ConceptNode clone = new ConceptNode(conceptNode);
    final ConceptNode clone1 = new ConceptNode(conceptNode);

    Cobweb.cobweb(conceptNode, root);
    Cobweb.cobweb(clone, root);
    Cobweb.cobweb(clone1, root);

    Assertions.assertEquals(3, root.getChildren().size());
    for (int i = 0; i < 3; ++i) {
      Assertions.assertTrue(root.getChildren().get(i).getChildren().isEmpty());
    }
  }


  /**
   * Comment.
   */
  @Disabled
  @Test
  void testMerge() {
    final ConceptNode root = new ConceptNode().root();
    final ConceptNode conceptNode = new ConceptNode();
    final NominalValue v = new NominalValue("test");
    final List<Value> val = new ArrayList<>();
    val.add(v);
    conceptNode.getAttributes().put("name", val);
    conceptNode.setId("a");
    final ConceptNode clone = new ConceptNode(conceptNode);
    final ConceptNode clone1 = new ConceptNode(conceptNode);

    final ConceptNode otherConcept = new ConceptNode();
    final NominalValue v1 = new NominalValue("other");
    final List<Value> val1 = new ArrayList<>();
    val1.add(v1);
    otherConcept.getAttributes().put("name", val1);
    otherConcept.setId("b");
    final ConceptNode other1 = new ConceptNode(otherConcept);

    Cobweb.cobweb(conceptNode, root);
    Cobweb.cobweb(clone, root);
    Cobweb.cobweb(otherConcept, root);
    Cobweb.cobweb(clone1, root);
    Cobweb.cobweb(other1, root);

    Assertions.assertEquals(2, root.getChildren().size());
    Assertions.assertEquals(3, root.getChildren().get(0).getChildren().size());
    Assertions.assertEquals(2, root.getChildren().get(1).getChildren().size());
  }
}
