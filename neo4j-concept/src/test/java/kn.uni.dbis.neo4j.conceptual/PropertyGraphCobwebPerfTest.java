package kn.uni.dbis.neo4j.conceptual;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.factory.GraphDatabaseSettings;
import org.neo4j.internal.kernel.api.exceptions.KernelException;
import org.neo4j.server.CommunityBootstrapper;

import kn.uni.dbis.neo4j.conceptual.algos.PropertyGraphCobweb;
import kn.uni.dbis.neo4j.conceptual.proc.PropertyGraphCobwebProc;
import kn.uni.dbis.neo4j.eval.DefaultPaths;
import kn.uni.dbis.neo4j.eval.TestDatabaseFactory;
import kn.uni.dbis.neo4j.eval.util.FileUtils;

/**
 * Utility class for executing the algorithm for profiling purposes.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 *
 */
public final class PropertyGraphCobwebPerfTest {
  /**
   * Hidden default constructor.
   */
  private PropertyGraphCobwebPerfTest() {
    // NOOP
  }

  /**
   * Runs the Algorithm without junit or jmh surrounding it for profiling.
   * @param args none
   * @throws IOException if copying the dataset fails
   */
  public static void main(final String[] args) throws IOException {
    final Path cloneFrom = DefaultPaths.PLAIN_STORES_PATH.resolve("Rome99.db");
    System.out.println(cloneFrom);

    // start a web GUI on a copy of our test graph
    // loaded with our procedures
    final CommunityBootstrapper cb = new CommunityBootstrapper();
    final File clone = new FileUtils().clone(cloneFrom, Paths.get(DefaultPaths.TEMP_HOME_PATH.toString())).toFile();

    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
      try {
        org.apache.commons.io.FileUtils.deleteDirectory(clone);
      } catch (final IOException e) {
        e.printStackTrace();
      }
    }));
    final Class[] procedures = {PropertyGraphCobweb.class};

    final Map<String, String> config = new HashMap<>();
    // assume folder has the same name as database
    config.put(GraphDatabaseSettings.procedure_unrestricted.name(), "*");
    config.put(GraphDatabaseSettings.active_database.name(), cloneFrom.getFileName().toString());
    config.put(GraphDatabaseSettings.plugin_dir.name(), Paths.get("target").toString());
    config.put(GraphDatabaseSettings.auth_enabled.name(), "false");

    cb.start(clone, Optional.empty(), config);
    final GraphDatabaseService db =
        cb.getServer().getDatabase().getGraph().getGraphDatabase();
    try {
      new TestDatabaseFactory().registerProcedure(db, procedures);
    } catch (final KernelException e) {
      e.printStackTrace();
    }

    try (Transaction tx = db.beginTx()) {
      final Stream<Node> nodes = db.getAllNodes().stream().limit(100);
      System.out.println(nodes.count());
      PropertyGraphCobwebProc proc = new PropertyGraphCobwebProc(db);
      final PropertyGraphCobweb tree = proc.integrate(db.getAllNodes().stream(),
          db.getAllRelationships().stream()).findFirst().orElseThrow(() -> new RuntimeException("Unreachable"));
      tree.prettyPrint();
    }
  }
}
