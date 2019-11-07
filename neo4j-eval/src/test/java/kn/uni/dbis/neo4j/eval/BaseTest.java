/*
 * @(#)BaseTest.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;

import kn.uni.dbis.neo4j.eval.annotations.GraphDBConfig;
import kn.uni.dbis.neo4j.eval.annotations.GraphDBSetup;
import kn.uni.dbis.neo4j.eval.annotations.GraphSource;
import kn.uni.dbis.neo4j.eval.annotations.Preprocessing;
import kn.uni.dbis.neo4j.eval.annotations.Procedures;
import kn.uni.dbis.neo4j.eval.datasets.Dataset;
import kn.uni.dbis.neo4j.eval.util.FileUtils;

// FIXME more setup tests

/**
 * Base Class for all test, intended to setup and check base functionality.
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@ExtendWith(GraphDBSetup.class)
class BaseTest {
  /**
   * Deletes all resource folders to prevent unnecessary disk usage.
   */
  @AfterAll
  static void clean() {
    try {
      new FileUtils().recursiveDeleteOnExit(DefaultPaths.PERSISTENT_HOME_PATH.getParent());
    } catch (final IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * @param db       database service instance to test the call on.
   * @param call     name of the procedure to call.
   * @param consumer Lambda expression to check the returned result
   */
  void testCallSingleResult(final GraphDatabaseService db, final String call, final Consumer<Map<String,
      Object>> consumer) {
    this.testResult(db, call, res -> {
      Assertions.assertTrue(res.hasNext(), "No result.");
      final Map<String, Object> row = res.next();
      consumer.accept(row);
      Assertions.assertFalse(res.hasNext(), "Unexpected extra result.");
    });
  }

  /**
   * @param db        database instance to execute the statement on
   * @param statement statement to execute
   * @param v         result visitor
   */
  void testCall(final GraphDatabaseService db, final String statement,
                       final Result.ResultVisitor<RuntimeException> v) {
    try (Transaction tx = db.beginTx()) {
      final Result res = db.execute(statement);
      res.accept(v);
      tx.success();
    }
  }

  /**
   * @param db             database instance to execute the statement on
   * @param call           call to execute
   * @param resultConsumer result consumer
   */
  private void testResult(final GraphDatabaseService db, final String call,
                                 final Consumer<Result> resultConsumer) {
    try (Transaction tx = db.beginTx()) {
      resultConsumer.accept(db.execute(call));
      tx.success();
    }
  }

  /**
   * Test weather instatiation with the defaults work.
   *
   * @param db to be tested
   */
  @Test
  @GraphDBConfig()
  @GraphSource()
  @Preprocessing()
  @Procedures()
  void testDefaultConfig(final GraphDatabaseService db) {
    // TODO find a useful test
    final Result result = db.execute("CALL dbms.procedures() "
        + "YIELD name "
        + "RETURN *");
    System.out.println(result.resultAsString());
    Assertions.assertTrue(Files.exists(DefaultPaths.PLAIN_STORES_PATH.resolve("EDBT17_RUNNING_EXAMPLE.db/neostore")));
  }

  /**
   * Test weather instatiation with the defaults work.
   *
   * @param db to be tested
   */
  @Test
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.EDBT17_RUNNING_EXAMPLE)
  @Preprocessing()
  @Procedures()
  void testCachedDefaultConfig(final GraphDatabaseService db) {
    final Result result = db.execute("CALL dbms.procedures() "
        + "YIELD name "
        + "RETURN *");
    System.out.println(result.resultAsString());
    Assertions.assertTrue(Files.exists(DefaultPaths.PLAIN_STORES_PATH.resolve("EDBT17_RUNNING_EXAMPLE.db/neostore")));
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.DBLP)
  @Preprocessing()
  @Procedures()
  void testDBLPDataset(final GraphDatabaseService db, final Dataset dataset) {

    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be used for testing
   * @param dataset to be used for testing
   */
  @Test
  @Disabled
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.LiveJournal1)
  @Preprocessing()
  @Procedures()
  void testLiveJournal1Dataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @Disabled
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.Orkut)
  @Preprocessing()
  @Procedures()
  void testOrkutDataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @Disabled
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.RoadNetCA)
  @Preprocessing()
  @Procedures()
  void testRoadNetCADataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.Rome99)
  @Preprocessing()
  @Procedures()
  void testRome99Dataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @Disabled
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.InternetTopology)
  @Preprocessing()
  @Procedures()
  void testInternetTopologyDataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @Disabled
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.RoadNetUSA)
  @Preprocessing()
  @Procedures()
  void testRoadNetUSADataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Test weather setting up a new template works (thus download, parse and import).
   *
   * @param db      to be tested
   * @param dataset to be used for testing
   */
  @Test
  @Disabled
  @GraphDBConfig()
  @GraphSource(getDataset = Dataset.RoadNetNY)
  @Preprocessing()
  @Procedures()
  void testRoadNetNYDataset(final GraphDatabaseService db, final Dataset dataset) {
    final Result nodeCount = db.execute("MATCH (n) RETURN count(n) as count");
    Assertions.assertEquals((long) dataset.getNodes(), nodeCount.next().values().toArray()[0]);

    final Result arcCount = db.execute("MATCH (n)-[r]->() RETURN count(r) ");
    Assertions.assertEquals((long) dataset.getArcs(), arcCount.next().values().toArray()[0]);
  }

  /**
   * Tests if the procedures to be tested are loaded into cypher.
   *
   * @param db              the database instance to which the procedures should be registered
   * @param procedurePrefix prefix of the name under which the procedures were registered
   */
  @Test
  @Disabled("META node not implemented")
  void testProceduresLoaded(final GraphDatabaseService db, final String procedurePrefix) {

    final AtomicInteger count = new AtomicInteger();

    this.testCall(db,
        "CALL dbms.procedures() "
            + "YIELD name "
            + "WHERE name STARTS WITH '" + procedurePrefix + "' "
            + "RETURN *",
        row -> {
          count.incrementAndGet();
          return true;
        });
    Assertions.assertEquals(2, count.get());
  }
}
