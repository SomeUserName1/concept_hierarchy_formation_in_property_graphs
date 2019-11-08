package kn.uni.dbis.neo4j.conceptual;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Transaction;

import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.ConceptValue;
import kn.uni.dbis.neo4j.conceptual.algos.NominalValue;
import kn.uni.dbis.neo4j.conceptual.algos.NumericValue;
import kn.uni.dbis.neo4j.conceptual.algos.PropertyGraphCobweb;
import kn.uni.dbis.neo4j.conceptual.algos.Value;
import kn.uni.dbis.neo4j.conceptual.proc.PropertyGraphCobwebProc;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBConfig;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBSetup;
import kn.uni.dbis.neo4j.eval.annotations.GraphSource;
import kn.uni.dbis.neo4j.eval.annotations.Procedures;

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
   */
  @Test
  void testCobweb(final GraphDatabaseService db) {
    try (Transaction tx = db.beginTx()) {
      final PropertyGraphCobweb tree = new PropertyGraphCobwebProc().integrate(db.getAllNodes().stream()).findFirst().
          orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);
      tree.print();
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

    System.out.println(c1);
    List<Value> vals =  c1.getAttributes().get("A");
    Assertions.assertEquals(2, vals.get(vals.indexOf(new NominalValue("a"))).getCount());
    vals =  c1.getAttributes().get("B");
    Assertions.assertEquals(2, vals.get(vals.indexOf(new NumericValue(1))).getCount());
    vals =  c1.getAttributes().get("C");
    Assertions.assertEquals(2, vals.get(vals.indexOf(cv)).getCount());
    System.out.println("============================= UpdateCounts OK ====================================");
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
    System.out.println("============================= Equal Methods OK ====================================");
  }
}
