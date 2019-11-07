/*
 * @(#)DefaultPaths.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Paths that are used in multiple places (annootations.GraphDBSetup, datasets, ...).
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class DefaultPaths {
  /**
   * Neo4J home Path for caching database files with plain datasets pre-imported.
   */
  public static final Path PLAIN_HOME_PATH = Paths.get("src/test/resources/plain");
  /**
   * Path for caching database files with plain datasets pre-imported.
   */
  public static final Path PLAIN_STORES_PATH = Paths.get(PLAIN_HOME_PATH + "/data/databases/");
  /**
   * Neo4J home Path for temporary database instance files and dataset sources.
   */
  public static final Path TEMP_HOME_PATH = Paths.get("src/test/resources/temp");
  /**
   * Path for temporary database instance files and dataset sources.
   */
  public static final Path TEMP_STORES_PATH = Paths.get(TEMP_HOME_PATH + "/data/databases/");
  /**
   * Neo4J home Path for persistent test database instance files to be able to inspect the results in more detail.
   */
  public static final Path PERSISTENT_HOME_PATH = Paths.get("src/test/resources/persist");
  /**
   * Path for persistent test database instance files to be able to inspect the results in more detail.
   */
  public static final Path PERSISTENT_STORES_PATH = Paths.get(PERSISTENT_HOME_PATH + "/data/databases/");
  /**
   * Hidden default constructor.
   */
  private DefaultPaths() {
    // Hidden default constructor
  }
}
