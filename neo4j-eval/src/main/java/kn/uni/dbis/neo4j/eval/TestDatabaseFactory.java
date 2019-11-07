/*
 * @(#)TestDatabaseFactory.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval;

import java.io.IOException;
import java.nio.file.Path;

import org.neo4j.graphdb.DependencyResolver;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.factory.GraphDatabaseBuilder;
import org.neo4j.graphdb.factory.GraphDatabaseFactory;
import org.neo4j.graphdb.factory.GraphDatabaseSettings;
import org.neo4j.internal.kernel.api.exceptions.KernelException;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.logging.slf4j.Slf4jLogProvider;

import apoc.export.graphml.ExportGraphML;

/**
 * Database factory utility for testing purposes.
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class TestDatabaseFactory {
  /**
   * Create a new embedded database at the given path.
   *
   * @param dbPath   path to database
   * @param log      Flag indicating weather to use a user provided logger
   * @param readOnly Flag indicating weather to open database read only
   * @return database
   */
  public GraphDatabaseService newEmbedded(final Path dbPath, final boolean log, final boolean readOnly) {
    final GraphDatabaseFactory factory = new GraphDatabaseFactory();
    if (log) {
      factory.setUserLogProvider(new Slf4jLogProvider());
    }
    final GraphDatabaseBuilder builder = factory.newEmbeddedDatabaseBuilder(dbPath.toFile());
    builder.setConfig(GraphDatabaseSettings.read_only, Boolean.toString(readOnly));
    builder.setConfig(GraphDatabaseSettings.keep_logical_logs, Boolean.toString(log));
    return builder.newGraphDatabase();
  }

  /**
   * Open an embedded database at the given path.
   *
   * @param existingDBDataDir path to neostore files
   * @param log               Flag indicating weather to use a user provided logger
   * @param readOnly          Flag indicating weather to open database read only
   * @return database
   * @throws IOException if the directory is non-existent or not a propper neostore
   */
  public GraphDatabaseService openEmbedded(final Path existingDBDataDir, final boolean log,
                                                  final boolean readOnly) throws IOException {
    if (!existingDBDataDir.toFile().exists()) {
      throw new IOException("Cannot open database at non-existing directory: " + existingDBDataDir);
    }
    if (!existingDBDataDir.resolve("neostore").toFile().exists()) {
      throw new IOException("Did not find neostore in given data directory: " + existingDBDataDir);
    }
    return this.newEmbedded(existingDBDataDir, log, readOnly);
  }

  /**
   * Export the graph as GraphML into the specified file.
   *
   * @param db         graph database, has to have {@code apoc.export.file.enabled=true} in config.
   * @param exportFile file to export into
   * @throws KernelException if the export procedure could not be registered in the database
   */
  public void exportGraphML(final GraphDatabaseService db, final Path exportFile) throws KernelException {
    // final Path export = Paths.get(dataset.dbName + ".graphml");
    this.registerProcedure(db, ExportGraphML.class);
    db.execute("call apoc.export.graphml.all(\"" + exportFile.toAbsolutePath().toString() + "\",{})\n");
  }

  /**
   * Registeres procedures to the Cypher interface of the given database instance.
   *
   * @param db         database instance to register the procedures to
   * @param procedures procedures to register
   * @throws KernelException throws if the registration failed
   */
  public void registerProcedure(final GraphDatabaseService db, final Class<?>... procedures) throws
      KernelException {
    final Procedures proceduresService =
        ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class,
            DependencyResolver.SelectionStrategy.ONLY);
    for (final Class<?> procedure : procedures) {
      // FIXME why all 3
      proceduresService.registerProcedure(procedure, true);
      proceduresService.registerFunction(procedure, true);
      proceduresService.registerAggregationFunction(procedure, true);
    }
  }
}
