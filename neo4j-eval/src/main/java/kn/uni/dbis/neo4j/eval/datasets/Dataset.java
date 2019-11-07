/*
 * @(#)Dataset.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval.datasets;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.RelationshipType;

import kn.uni.dbis.neo4j.eval.DefaultPaths;
import kn.uni.dbis.neo4j.eval.TestDatabaseFactory;

/**
 * Common interface for all datasets.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public enum Dataset {
  /**
   * DBLP dataset metadata.
   */
  DBLP("CO-AUTHOR", "https://snap.stanford"
      + ".edu/data/bigdata/communities/com-dblp.ungraph.txt.gz", false, false,
      Dataset.SourceType.SNAP, 317080, 1049866),

  /**
   * Live Journal dataset metadata.
   */
  LiveJournal1("FRIENDS", "https://snap.stanford"
      + ".edu/data/bigdata/communities/com-lj.ungraph.txt.gz", false, false, SourceType.SNAP, 3997962,
      34681189),

  /**
   * Orkut dataset metadata.
   */
  Orkut("FRIENDS",
      "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz", false, false,
      SourceType.SNAP, 3072441, 117185083),

  /**
   * RoadNetCA dataset metadata.
   */
  RoadNetCA("ROAD", "https://snap.stanford.edu/data/roadNet-CA.txt.gz", false,
      false, SourceType.SNAP, 1965206, 5533214),

  /**
   * Rome99 dataset metadata.
   */
  Rome99("ROAD", "http://users.diag.uniroma1.it/challenge9/data/rome/rome99"
      + ".gr", true, true, SourceType.Challenge9, 3353, 8870),

  /**
   * RoadNetUSA dataset meta information.
   */
  RoadNetUSA("ROAD",
      "http://users.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz", false, true,
      SourceType.Challenge9, 23947347, 58333344),
  /**
   * RoadNetNY dataset meta information.
   */
  RoadNetNY("ROAD",
      "http://users.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz", false, true,
      SourceType.Challenge9, 264346, 733846),

  // TODO Add more from here: http://users.diag.uniroma1.it/challenge9/download.shtml

  /**
   * Internet Topology dataset metadata.
   */
  InternetTopology("CONNECTED_TO", "https://snap.stanford.edu/data/as-skitter"
      + ".txt.gz", false, false, SourceType.SNAP, 1696415, 11095298),

  /**
   * Handcrafted dataset metadata.
   */
  EDBT17_RUNNING_EXAMPLE("WAY", "CREATE\n"
      + "(n0 {name:\"n0\"}),\n"
      + "(n1 {name:\"n1\"}),\n"
      + "(n2 {name:\"n2\"}),\n"
      + "(n3 {name:\"n3\"}),\n"
      + "(n4 {name:\"n4\"}),\n"
      + "(n5 {name:\"n5\"}),\n"
      + "(n6 {name:\"n6\"}),\n"
      + "(n0)-[:WAY { weight:6 }]->(n1),\n"
      + "(n0)-[:WAY { weight:4 }]->(n2),\n"
      + "(n0)-[:WAY { weight:3 }]->(n3),\n"
      + "(n1)-[:WAY { weight:6 }]->(n0),\n"
      + "(n1)-[:WAY { weight:2 }]->(n3),\n"
      + "(n1)-[:WAY { weight:6 }]->(n6),\n"
      + "(n2)-[:WAY { weight:4 }]->(n0),\n"
      + "(n2)-[:WAY { weight:3 }]->(n3),\n"
      + "(n2)-[:WAY { weight:5 }]->(n4),\n"
      + "(n3)-[:WAY { weight:3 }]->(n0),\n"
      + "(n3)-[:WAY { weight:2 }]->(n1),\n"
      + "(n3)-[:WAY { weight:3 }]->(n2),\n"
      + "(n3)-[:WAY { weight:5 }]->(n4),\n"
      + "(n3)-[:WAY { weight:3 }]->(n5),\n"
      + "(n4)-[:WAY { weight:5 }]->(n2),\n"
      + "(n4)-[:WAY { weight:5 }]->(n3),\n"
      + "(n4)-[:WAY { weight:1 }]->(n5),\n"
      + "(n4)-[:WAY { weight:2 }]->(n6),\n"
      + "(n5)-[:WAY { weight:3 }]->(n3),\n"
      + "(n5)-[:WAY { weight:1 }]->(n4),\n"
      + "(n5)-[:WAY { weight:2 }]->(n6),\n"
      + "(n6)-[:WAY { weight:6 }]->(n1),\n"
      + "(n6)-[:WAY { weight:2 }]->(n4),\n"
      + "(n6)-[:WAY { weight:2 }]->(n5)", true, true, SourceType.CypherQuery, 7, 24);

  /**
   * Flag indicating wether the Dataset is based on a query.
   */
  private final SourceType srcType;
  /**
   * Name of the relationship type.
   */
  private String relationshipTypeName;
  /**
   * the source url of the dataset or the Cypher Query String.
   */
  private String source;
  /**
   * path where the dataset is downloaded to.
   */
  private Path path;
  /**
   * flag indicating if the arcs are directed.
   */
  private boolean directed;
  /**
   * flag indicating wether the graph is weighted.
   */
  private boolean weighted;
  /**
   * number of nodes.
   */
  private int noNodes;
  /**
   * number of edges.
   */
  private int noArcs;
  /**
   * Snap/Challenge9 dataset constructor.
   *
   * @param relationshipTypeName name of the relationship
   * @param source               url where one can download the dataset
   * @param directed             if the graphs edges are to be interpreted directed
   * @param weighted             wether the graph contains weighted edges
   * @param srcType              Type of the source so either a handcrafted query or a specific source listed in
   *                             {@link SourceType}
   * @param noNodes              number of nodes in the data set
   * @param noArcs               number of Arcs in the dataset
   */
  Dataset(final String relationshipTypeName, final String source, final boolean directed, final boolean weighted,
          final SourceType srcType, final int noNodes, final int noArcs) {
    this.relationshipTypeName = relationshipTypeName;
    this.source = source;
    this.directed = directed;
    this.weighted = weighted;
    this.srcType = srcType;
    this.noNodes = noNodes;
    this.noArcs = noArcs;
  }

  /**
   * Constructor used to define an existing database to open for testing.
   *
   * @param path path to the neo4j home directory of the database
   */
  Dataset(final Path path) {
    this.path = path;
    this.srcType = SourceType.Custom;
  }

  /**
   * Accessor method for the directed flag.
   *
   * @return if the edges are to be interpreted as directed in the procedure/query
   */
  public boolean isDirected() {
    return this.directed;
  }

  /**
   * Accessor method for the weighted flag.
   *
   * @return if the edges are to be interpreted as weighted in the procedure/query
   */
  public boolean isWeighted() {
    return this.weighted;
  }

  /**
   * Accessor method for number of nodes.
   *
   * @return the number of nodes in the dataset
   */
  public int getNodes() {
    return this.noNodes;
  }

  /**
   * Accessor method for number of nodes.
   *
   * @return the number of nodes in the dataset
   */
  public int getArcs() {
    return this.noArcs;
  }

  /**
   * Accessor method for the path where the dataset is to be saved to.
   *
   * @return the {@link Path} where the dataset goes
   */
  public Path getPath() {
    return this.path;
  }

  /**
   * Returns the relationship type.
   *
   * @return RelationshipType of the Neo4J Relationship
   */
  public RelationshipType getRelationshipType() {
    return RelationshipType.withName(this.relationshipTypeName);
  }

  /**
   * returns the url of the datasets if available online, if it's a handcrafted query, returns the query.
   *
   * @return a String that is either the url if SourceType is not CypherQuery or the quesry if the source type is
   * Cypher query
   */
  public String getSource() {
    return this.source;
  }

  /**
   * Accessor method for the {@link SourceType} of the dataset.
   *
   * @return the {@link SourceType} of the dataset
   */
  public SourceType getSrcType() {
    return this.srcType;
  }

  /**
   * Entrance point for data download, parsing and import. Used in @link{annotations.GraphDBSetup} to ensure a
   * neo store exists that holds the plain dataset.
   * When adding another @link{SourceType} one needs to add methods for the tasks mentioned above and hook them here.
   */
  public void fetchAndImport() {
    switch (this.srcType) {
      case SNAP:
      case Challenge9:
        this.path = DefaultPaths.TEMP_HOME_PATH.resolve(this.name());
        System.out.println(this.path);
        try {
          new DatasetInitializer().downloadAndParseDataset(this);
          new CSVImporter().csvImport(this);
        } catch (final IOException e) {
          e.printStackTrace();
        }
        break;
      case CypherQuery:
        final TestDatabaseFactory tdf = new TestDatabaseFactory();
        final GraphDatabaseService db = tdf.newEmbedded(
            Paths.get(DefaultPaths.PLAIN_STORES_PATH.toString(), this.name() + ".db"),
            false, false);
        db.execute(this.source).close();
        break;
      default:
        throw new IllegalStateException("Unreachable");
    }
    System.gc();
  }

  /**
   * Types of currently supported dataset sources.
   */
  public enum SourceType {
    /**
     * Dataset of the stanford network analytics platform project in the txt.gz format (carefull not all SNAP datasets
     * are in the same format).
     */
    SNAP,
    /**
     * Dataset from DIMACS Challenge9.
     */
    Challenge9,
    /**
     * A Cypher query is used as dataset.
     */
    CypherQuery,
    /**
     * Existing neo4j store with customizations.
     */
    Custom
  }
}
