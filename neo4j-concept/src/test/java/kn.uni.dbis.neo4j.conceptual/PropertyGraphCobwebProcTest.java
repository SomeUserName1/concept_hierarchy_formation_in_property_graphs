package kn.uni.dbis.neo4j.conceptual;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import kn.uni.dbis.neo4j.conceptual.algos.ConceptValue;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;

import kn.uni.dbis.neo4j.conceptual.algos.Cobweb;
import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.NominalValue;
import kn.uni.dbis.neo4j.conceptual.algos.NumericValue;
import kn.uni.dbis.neo4j.conceptual.algos.PropertyGraphCobweb;
import kn.uni.dbis.neo4j.conceptual.algos.Value;
import kn.uni.dbis.neo4j.conceptual.proc.PropertyGraphCobwebProc;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBConfig;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBSetup;
import kn.uni.dbis.neo4j.eval.annotations.GraphSource;
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
  @Test
  void testCobwebSmall(final GraphDatabaseService db, final Dataset dataset) {
    try (Transaction tx = db.beginTx()) {
      final Stream<Node> nodes = db.getAllNodes().stream();
      System.out.println(nodes.count());
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream(),
          db.getAllRelationships().stream()).findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);
      tree.print();

      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
          tree.getNodeSummaryTree()};
      for (ConceptNode root : subtrees) {
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), dataset.getNodes());
      Assertions.assertEquals(this.leafCount(subtrees[1]), dataset.getArcs());
      Assertions.assertEquals(this.leafCount(subtrees[2]), dataset.getNodes());
    }
  }

  /**
   * Tests if the procedure throws runtime errors and if the resulting tree is returned.
   *
   * @param db database to execute the procedure call against
   * @param dataset the dataset specified in the class scope, used to access it's node and relationship count
   */
  @Test
  @GraphSource(getDataset = Dataset.Rome99)
  void testCobwebMedium(final GraphDatabaseService db, final Dataset dataset) {
    try (Transaction tx = db.beginTx()) {
      final PropertyGraphCobweb tree = PropertyGraphCobwebProc.integrate(db.getAllNodes().stream(),
          db.getAllRelationships().stream()).findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);

      // FIXME assetion fails on count :/
      // TODO fix count
      final ConceptNode[] subtrees = {tree.getNodePropertiesTree(), tree.getRelationshipPropertiesTree(),
        tree.getNodeSummaryTree()};
      for (ConceptNode root : subtrees) {
        // subtree.print();
        this.checkPartitionCounts(root);
        this.checkParent(root);
        this.checkLeafType(root);
      }
      Assertions.assertEquals(this.leafCount(subtrees[0]), dataset.getNodes());
      Assertions.assertEquals(this.leafCount(subtrees[1]), dataset.getArcs());
      Assertions.assertEquals(this.leafCount(subtrees[2]), dataset.getNodes());
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
    Assertions.assertEquals(2, vals.get(vals.indexOf(new NumericValue(1))).getCount());
    vals =  c1.getAttributes().get("C");
    Assertions.assertEquals(2, vals.get(vals.indexOf(cv)).getCount());
  }

  /**
   * Test equals methods.
   */
  @Test
  void testEquals() {
    final NominalValue n1 = new NominalValue("a");
    final NominalValue n2 = new NominalValue("b");
    final NominalValue n3 = new NominalValue("a");

    Assertions.assertEquals(n1, n3);
    Assertions.assertNotEquals(n1, n2);

    final NumericValue n4 = new NumericValue(1);
    final NumericValue n5 = new NumericValue(2);
    final NumericValue n6 = new NumericValue(1);

    Assertions.assertEquals(n4, n6);
    Assertions.assertNotEquals(n4, n5);

    final ConceptNode n7 = new ConceptNode();
    final ConceptNode n8 = new ConceptNode();
    final ConceptNode n9 = new ConceptNode();
    final ConceptNode n10 = new ConceptNode();


    final List<Value> nomList = new ArrayList<>();
    nomList.add(new NominalValue("a"));
    n7.getAttributes().put("A", nomList);
    n8.getAttributes().put("A", nomList);

    Assertions.assertEquals(n7, n8);

    final List<Value> numList = new ArrayList<>();
    numList.add(new NumericValue(1));
    n9.getAttributes().put("B", numList);
    n10.getAttributes().put("A", numList);

    Assertions.assertNotEquals(n7, n9);
    Assertions.assertNotEquals(n7, n10);
    Assertions.assertNotEquals(n9, n10);


    final ConceptValue c1 = new ConceptValue(new ConceptNode());
    final ConceptValue c2 = new ConceptValue(new ConceptNode());
    final ConceptValue c3 = new ConceptValue(n7);

    Assertions.assertEquals(c1, c2);
    Assertions.assertNotEquals(c1, c3);
  }

  /**
   * Test if nodes are created correctly.
   */
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
  @Test
  @Disabled
  void testSplit() {
  }

  /**
   * Comment.
   */
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
