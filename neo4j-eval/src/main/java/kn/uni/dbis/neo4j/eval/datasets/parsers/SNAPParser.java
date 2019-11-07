/*
 * @(#)SNAPParser.java   1.0   May 15, 2019
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
import java.util.HashSet;

import kn.uni.dbis.neo4j.eval.datasets.Dataset;

/**
 * Contains parser method for .gr files (DIMACS Challenge 9 format).
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class SNAPParser {
  /**
   * parses ONLY files in txt.gz format with schema "From To Weight" from the SNAP datasets.
   *
   * @param br       reads the downloaded file to be parsed
   * @param outNodes compressed File output stream for the nodes
   * @param outArcs  as above just for the arcs
   * @param dataset  dataset instance to parse, necessary for relationship name
   * @throws IOException if reading or writing fails
   */
  public void parseSNAP(final BufferedReader br, final BufferedWriter outNodes, final BufferedWriter outArcs,
                               final Dataset dataset)
      throws IOException {

    int flushCount = 0;
    final HashSet<String> nodeIDs = new HashSet<>();
    final StringBuilder sb = new StringBuilder();
    String line = br.readLine();
    String[] lineSplit;
    while (line != null) {
      lineSplit = line.trim().split("\\s+");

      if (lineSplit[0].equals("#")) {
        line = br.readLine();
        continue;
      }

      // nodes need to be added to the set until the last line was read to avoid duplicate entries
      nodeIDs.add(lineSplit[0]);
      nodeIDs.add(lineSplit[1]);

      // write & flush the arcs only every 100 lines to reduce IO overhead
      sb.append(String.join(", ", lineSplit)).append(", ")
          .append(dataset.getRelationshipType().name()).append("\n");
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
    outNodes.write(String.join("\n", nodeIDs.toArray(new String[]{})));
    outNodes.flush();
  }
}
