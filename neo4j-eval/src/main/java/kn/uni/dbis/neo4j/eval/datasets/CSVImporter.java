/*
 * @(#)CSVImporter.java   1.0   May 15, 2019
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
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;

import kn.uni.dbis.neo4j.eval.DefaultPaths;

/**
 * Calls the neo4j-admin tool to batch import a dataset, currently only those which are in format of only 2 files.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
final class CSVImporter {
  // TODO support multiple files?
  /**
   * import a dataset into the database using the admin tool bulk import FOR ONE NODES and ONE Relationships file
   * named .
   *
   *
   + "dbms.jvm.additional=-XX:+UseG1GC\np"
   + "dbms.jvm.additional=-XX:+AlwaysPreTouch\n"
   + "dbms.jvm.additional=-XX:+UnlockExperimentalVMOptions\n"
   + "dbms.jvm.additional=-XX:+TrustFinalNonStaticFields\n"
   *
   * @param dataset the dataset that shall be imported
   * @throws IOException if the config file couldn't be created or if already one existed
   */
  void csvImport(final Dataset dataset) throws IOException {
    final String conf = "dbms.active_database=" + dataset.name() + ".db\n"
        + "dbms.directories.data=" + DefaultPaths.PLAIN_STORES_PATH.getParent().toAbsolutePath().toString() + "\n"
        + "dbms.directories.logs=" + DefaultPaths.PLAIN_STORES_PATH.resolve("logs").toAbsolutePath().toString()+ "\n"
        + "dbms.memory.heap.initial_size=2048m\n"
        + "dbms.memory.heap.max_size=2048\n";

    final File confFile = DefaultPaths.TEMP_HOME_PATH.resolve("import_tool.conf").toFile();
    if (!confFile.createNewFile()) {
      throw new IOException("Conf file already exists!");
    } else {
      try (BufferedWriter outConf = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(confFile),
          StandardCharsets.UTF_8))) {
        outConf.write(conf);
        outConf.flush();
      }
    }

    final String arcsData = Paths.get(dataset.getPath().toString() + "-arcs.csv").toAbsolutePath().toString();
    final String nodesData = Paths.get(dataset.getPath().toString() + "-nodes.csv").toAbsolutePath().toString();

    final String relationshipName = dataset.getRelationshipType().name();

    final String[] command =
        ("neo4j-admin import --additional-config=" + confFile.toPath().toAbsolutePath().toString()
            + " --mode=csv --database=" + dataset.name() + ".db "
            + "--id-type=STRING " + " --max-memory=2G " + "--nodes " + String.format("%s ", nodesData)
            + "--relationships:" + relationshipName + String.format(" %s", arcsData)).split(" ");
    try {
      System.gc();
      System.out.println("Starting import");
      System.out.println(Arrays.toString(command));
      final ProcessBuilder pb = new ProcessBuilder(command);
      pb.environment().put("HEAP_SIZE", "1152m");
      pb.environment().put("JAVA_OPTS", "-Xms1152m -Xmx1152m");
      pb.directory(DefaultPaths.PLAIN_HOME_PATH.toRealPath().toFile());
      final Process importTool = pb.start();
      final InputStreamReader inputStreamReaderErr = new InputStreamReader(importTool.getErrorStream(),
          StandardCharsets.UTF_8);
      final InputStreamReader inputStreamReaderIn = new InputStreamReader(importTool.getInputStream(),
          StandardCharsets.UTF_8);
      final BufferedReader ebr = new BufferedReader(inputStreamReaderErr);
      final BufferedReader ibr = new BufferedReader(inputStreamReaderIn);

      this.printBuffer(ebr);
      this.printBuffer(ibr);

      importTool.waitFor();

      this.printBuffer(ebr);
      this.printBuffer(ibr);

      if (!confFile.delete()) {
        System.out.println("Couldn't delete Import tool conf file!");
      }
      inputStreamReaderErr.close();
      inputStreamReaderIn.close();

      ebr.close();
      ibr.close();

      System.gc();
    } catch (final InterruptedException | IOException e) {
      e.printStackTrace();
    }
    System.out.println("finished import");
  }

  /**
   * prints the content of a buffered reader.
   *
   * @param br bufferedreader to print
   */
  private void printBuffer(final BufferedReader br) {
    String readLine;

    try {
      readLine = br.readLine();
      while (readLine != null) {
        System.out.println(readLine);

        readLine = br.readLine();
      }
    } catch (final IOException e) {
      e.printStackTrace();
    }
  }
}
