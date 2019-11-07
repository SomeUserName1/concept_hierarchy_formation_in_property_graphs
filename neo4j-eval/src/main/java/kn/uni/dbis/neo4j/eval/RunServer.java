/*
 * @(#)RunServer.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.factory.GraphDatabaseSettings;
import org.neo4j.internal.kernel.api.exceptions.KernelException;
import org.neo4j.logging.FormattedLogProvider;
import org.neo4j.logging.Log;
import org.neo4j.server.CommunityBootstrapper;

import kn.uni.dbis.neo4j.eval.util.FileUtils;

/**
 * Spawn a server using a clone of a test database.
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
final class RunServer {
  /**
   * Hidden default constructor.
   */
  private RunServer() {
  }

  /**
   * Entry point to run a database for development.
   *
   * @param args first arg: the name of the database to be tested on.
   *             all other args: Custom procedures to be hooked and tested
   *             with  the db server
   * @throws IOException if the copied database file couldn't be removed
   */
  public static void main(final String... args) throws IOException {
    final Log mLogger = FormattedLogProvider.toOutputStream(System.out).getLog(RunServer.class);
    if (!(args.length > 2)) {
      System.out.println("This method takes 2 or more arguments: \n"
          + "1. the name of the test database to be used, specified without "
          + "the .db\n "
          + "2. The procedure classes that shall be hooked to cypher.\n");
      return;
    }
    final String testDatabaseName = args[0];
    final Class<?>[] procedures = new Class<?>[args.length - 1];
    for (int i = 1; i < args.length; ++i) {
      try {
        procedures[i - 1] = Class.forName(args[i]);
      } catch (final ClassNotFoundException e) {
        e.printStackTrace();
      }
    }


    final Path cloneFrom = DefaultPaths.PLAIN_STORES_PATH.resolve(testDatabaseName + ".db");

    // start a web GUI on a copy of our test graph
    // loaded with our procedures
    final CommunityBootstrapper cb = new CommunityBootstrapper();
    final File clone = new FileUtils().clone(cloneFrom, Paths.get(DefaultPaths.TEMP_HOME_PATH.toString())).toFile();

    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
      try {
        org.apache.commons.io.FileUtils.deleteDirectory(clone);
      } catch (final IOException e) {
        mLogger.error("Could not delete cloned database " + e.getLocalizedMessage());
      }
    }));

    mLogger.info("Opening (cloned) database at " + clone + " (with deleteOnExit)");

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
    } catch (final KernelException ke) {
      mLogger.error("KernelException occured when registering the procedure"
      );
      mLogger.error(ke.getLocalizedMessage());
    }
  }
}
