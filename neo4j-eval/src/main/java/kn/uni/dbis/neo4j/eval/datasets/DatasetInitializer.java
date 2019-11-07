/*
 * @(#)DatasetInitializer.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval.datasets;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

import kn.uni.dbis.neo4j.eval.datasets.Dataset.SourceType;
import kn.uni.dbis.neo4j.eval.datasets.parsers.Challenge9Parser;
import kn.uni.dbis.neo4j.eval.datasets.parsers.SNAPParser;

/**
 * Opens a single file defined as a string of an url that is either in plain text or gz format.
 * Then starts writing out the header information and calls the appropriate parser.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
final class DatasetInitializer {
  /**
   * downloads the dataset from the source url if it's not a handwritten query.
   *
   * @param dataset The dataset instance that shall be parsed.
   * @throws IOException if creating the new file, or opening the buffered reader/writers fail
   */
  void downloadAndParseDataset(final Dataset dataset) throws IOException {
    if (dataset.getSrcType().equals(SourceType.CypherQuery)) {
      return;
    }
    final Path path = dataset.getPath();
    if (!path.getParent().toFile().exists()
        && !path.getParent().toFile().mkdirs()) {
      //throw new IOException("Couldn't create the directory structure!");
      System.out.println("Didnt create file structure");
      System.out.println(path.getParent().getParent().toFile());
    }
    final File nodesFile = new File(path.toString() + "-nodes.csv");
    final File arcsFile = new File(path.toString() + "-arcs.csv");
    if (!nodesFile.createNewFile() && !arcsFile.createNewFile()) {
      throw new IOException("Nodes and Arcs file already exist!");
    }

    final String source = dataset.getSource();
    try (
        BufferedReader br = source.contains("gz")
            ? new BufferedReader(new InputStreamReader(new GZIPInputStream(new URL(source).openStream()),
            StandardCharsets.UTF_8))
            : new BufferedReader(new InputStreamReader(new URL(source).openStream(),
            StandardCharsets.UTF_8));

        BufferedWriter outNodes = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(nodesFile),
            StandardCharsets.UTF_8));

        BufferedWriter outArcs = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(arcsFile),
            StandardCharsets.UTF_8))
    ) {

      outNodes.write("nodeId:ID\n");
      final String arcHeader = dataset.isWeighted() ? ":START_ID, :END_ID, weight:float, :TYPE\n" : ":START_ID,"
          + " :END_ID, :TYPE\n";
      outArcs.write(arcHeader);
      outNodes.flush();
      outArcs.flush();
      System.out.println("staring to parse");
      switch (dataset.getSrcType()) {
        case SNAP:
          new SNAPParser().parseSNAP(br, outNodes, outArcs, dataset);
          break;
        case Challenge9:
          new Challenge9Parser().parseChallenge9(br, outNodes, outArcs, dataset);
          break;
        default:
          throw new IllegalStateException("unreachable");
      }
      System.out.println("finished parsing");
    }
  }
}
