package kn.uni.dbis.neo4j.conceptual;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Transaction;

import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.NominalValue;
import kn.uni.dbis.neo4j.conceptual.algos.Value;
import kn.uni.dbis.neo4j.conceptual.algos.PropertyGraphCobweb;
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
  @Disabled
  void testCobweb(final GraphDatabaseService db) {
    try (Transaction tx = db.beginTx()) {
      final PropertyGraphCobweb tree = new PropertyGraphCobwebProc().integrate(db.getAllNodes().stream()).findFirst().
          orElseThrow(() -> new RuntimeException("Unreachable"));
      Assertions.assertNotNull(tree);
      tree.print();
    }
  }

  /**
   * JUST DO IT.
   */
  @Test
  void testUpdate() {
    final Map<Value, Integer> map = new HashMap<>();
    map.put(new NominalValue("a"), 1);
    final ConceptNode c1 = new ConceptNode();
    c1.getAttributes().put("A", map);
    final ConceptNode c2 = new ConceptNode();
    c2.getAttributes().put("A", map);

    c1.updateCounts(c2, true);

    System.out.println(c1);
    Assertions.assertEquals(2, (int) c1.getAttributes().get("A").get(new NominalValue("a")));
  }
}
