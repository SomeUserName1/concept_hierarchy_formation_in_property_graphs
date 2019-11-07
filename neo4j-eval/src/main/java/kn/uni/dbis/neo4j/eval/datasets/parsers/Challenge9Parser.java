/*
 * @(#)Challenge9Parser.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval.datasets.parsers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;

import kn.uni.dbis.neo4j.eval.datasets.Dataset;

/**
 * Contains parser method for .gr files (DIMACS Challenge 9 format).
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class Challenge9Parser {
  /**
   * Parse a .gr.gz file opened in a BufferedReader in DIMACS Challenge9 format to csv files for nodes and arcs
   * respectively.
   *
   * @param br       buffered reader containing the .gr.gz file
   * @param outNodes buffered writer, writing to the output-nodes.csv.gz file
   * @param outArcs  buffered writer, writing to the output-arcs.csv.gz file
   * @param dataset  dataset instance to parse, necessary for relationship name
   * @throws IOException if reading or writing fails
   */
  public void parseChallenge9(final BufferedReader br, final BufferedWriter outNodes,
                                     final BufferedWriter outArcs, final Dataset dataset)
      throws IOException {

    int flushCount = 0;
    final HashSet<String> nodeIDs = new HashSet<>();
    final StringBuilder sb = new StringBuilder();
    String line = br.readLine();
    String[] lineSplit;
    while (line != null) {
      lineSplit = line.trim().split("\\s+");
      if (!lineSplit[0].equals("a")) {
        line = br.readLine();
        continue;
      }
      // nodes need to be added to the set until the last line was read to avoid duplicate entries
      nodeIDs.add(lineSplit[1]);
      nodeIDs.add(lineSplit[2]);

      // flush the arcs only every 30 lines to reduce IO overhead
      sb.append(String.join(", ", Arrays.asList(lineSplit).subList(1, lineSplit.length)))
          .append(", ").append(dataset.getRelationshipType().name()).append("\n");

      if (flushCount >= 100) {
        outArcs.write(sb.toString());
        outArcs.flush();
        sb.setLength(0);
        flushCount = 0;
      }
      flushCount++;
      line = br.readLine();
    }
    outArcs.write(sb.toString());
    outArcs.flush();
    final String nodeOutStr = String.join("\n", nodeIDs.toArray(new String[]{}));
    outNodes.write(nodeOutStr);
    outNodes.flush();
  }
}
