package kn.uni.dbis.neo4j.conceptual.proc;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import kn.uni.dbis.neo4j.conceptual.algos.PropertyGraphCobweb;

/**
 * APOC hook for the PropertyGraphCobweb clustering Algorithm.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class PropertyGraphCobwebProc {
  /**
   * The database service to execute the procedure against.
   */
  @Context
  public GraphDatabaseService db;

  /**
   * @param nodes the node to be integrated
   * @return the root of the hierarchy
   */
  @Procedure(
      name = "kn.uni.dbis.neo4j.conceptual.PropertyGraphCobwebStream",
      mode = Mode.READ
  )
  public static Stream<PropertyGraphCobweb> integrate(@Name("nodes") final Stream<Node> nodes,
                                                      @Name("edges") final Stream<Relationship> relationships) {
    final PropertyGraphCobweb tree = new PropertyGraphCobweb();
    List<Node> nodesList = nodes.collect(Collectors.toList());
    List<Relationship> relationshipsList = relationships.collect(Collectors.toList());
    System.out.println("Number of nodes " + nodesList.size());
    System.out.println("Number of Relationships " + relationshipsList.size());
    tree.integrate(nodesList, relationshipsList);
    return Stream.of(tree);
  }
}
