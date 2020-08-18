package kn.uni.dbis.neo4j.conceptual.util;

import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.Value;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

/**
 * Utilities for trees.
 *  @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class TreeUtils {
  /** Logger. */
  private static final Logger LOG = Logger.getLogger("PropertyGraphCobweb");

  /**
   * Hidden default constructor.
   */
  private TreeUtils() {
    // NOOP
  }

  /**
   * computes the maximal depth of the tree.
   * @param node the currently visited node
   * @return an integer representing the depth of the tree
   */
  public static int deepestLevel(final ConceptNode node) {
    if (node.getChildren().isEmpty()) {
      return 1;
    } else {
      int deepest = 0;
      int temp;
      for (ConceptNode child : node.getChildren()) {
        temp = deepestLevel(child);
        if (temp > deepest) {
          deepest = temp;
        }
      }
      return deepest + 1;
    }
  }

  /**
   * find and return the conceptnode for already incorporated conceptnodes, else returns null.
   * @param id the id of the relationship.
   * @param node the node we are currently inspecting in the relationshipPropertiesCobweb tree
   * @return the corresponding conceptnode or null
   */
  public static ConceptNode findById(final String id, final ConceptNode node) {
    if (node.getId() != null) {
      return node.getId().equals(id) ? node : null;
    } else {
      ConceptNode temp;
      for (ConceptNode child : node.getChildren()) {
        temp = findById(id, child);
        if (temp != null) {
          return temp;
        }
      }
      return null;
    }
  }

  /**
   * Assigns a label to each node in the tree.
   * @param node currently visited node
   * @param parentLabel prefix of the current label
   * @param num postfix of the current label
   */
  public static void labelTree(final ConceptNode node, final String parentLabel, final String num) {
    node.setLabel(parentLabel + num);

    int i = 0;
    for (ConceptNode child : node.getChildren()) {
      labelTree(child, parentLabel + num, Integer.toString(i));
      i++;
    }
  }

  /**
   * Prints nodes recursively from node node downwards the tree.
   *
   * @param node node to print
   * @param sb    StringBuilder to use.
   * @param depth the depth when called in order to arrange the output appropriately
   * @param maxDepth the maximal depth to be considered
   * @return a String holding the representation of the tree
   */
  private static String printRec(final ConceptNode node, final StringBuilder sb, final int depth, final int maxDepth) {
    if (depth == 0) {
      sb.append("|__");
    } else {
      sb.append("\t".repeat(Math.max(0, depth)));
      sb.append("|____");
    }

    sb.append(node.toString()).append("\n");
    if (depth <= maxDepth) {
    final List<ConceptNode> localChildren;
      localChildren = new ArrayList<>(node.getChildren());

    for (ConceptNode child : localChildren) {
      printRec(child, sb, depth + 1, maxDepth);
      }
    }
    return sb.toString();
  }

  private static String getTexTables(final ConceptNode root, final int maxDepth) {
    TreeUtils.labelTree(root, "", "l");
    final StringBuilder sb = new StringBuilder();
    printRecTexTable(root, sb, 0, maxDepth);
    return sb.toString();
  }

  private static void printRecTexTable(final ConceptNode node, final StringBuilder sb, final int depth, final int maxDepth) {
    if (depth <= maxDepth) {
      toTexTable(node, sb);
      for (ConceptNode child : node.getChildren()) {
        if (child.getCount() / (double) node.getCount() > 0.05) {
          printRecTexTable(child, sb, depth + 1, maxDepth);
        }
      }
    }
  }


  private static void toTexTable(final ConceptNode node, StringBuilder sb) {
    sb.append("\n \n")
        .append("\\begin{table}[h] \n  \\centering \n \\begin{longtable}{|c|c|c|c|c|} \\hline \n")
        .append("Attribute & ValueType & Value & Probability & Occurences \\\\ \\hline \n");
    for (ConcurrentMap.Entry<String, List<Value>> attribute : node.getAttributes().entrySet()) {
      if (attribute.getValue().size() < 16) {
        sb.append("\\multirow{").append(attribute.getValue().size()).append("}{*}{").append(attribute.getKey())
            .append("}");
        List<Value> values = attribute.getValue();
        Value value;
        for (int i = 0; i < values.size(); ++i) {
          value = values.get(i);
          double prob = (double) value.getCount() / (double) node.getCount();
          prob = prob > 1 ? 1 : prob;
          sb.append(" & ").append(value.toTexString()).append("$")
              .append(String.format(Locale.US, "%.4f", prob)).append("$ & $").append(value.getCount()).append("$ ");
          if (i < values.size() - 1 ) {
            sb.append("\\\\ \\cline{2-5} \n");
          } else {
            sb.append("\\\\ \\hline \n");
          }
        }
      } else {
        String valueType = attribute.getValue().get(0).toTexString().startsWith("Num") ? "Numeric" : "Nominal";
        sb.append(attribute.getKey()).append(" & ").append(valueType).append(" & Too many values to display & & ")
            .append("\\\\ \\hline\n");
      }
    }
    sb.append("\\caption{").append("ConceptNode ").append(node.getLabel()).append(", P(node) = ")
        .append((double) node.getCount() / (double) node.getParent().getCount()).append(", Count ")
        .append(node.getCount()).append("}\n");
    sb.append("\\end{longtable}\n \\end{table} \n").append("\n");
  }


  private static String getTexForrest(final ConceptNode root, final int maxDepth) {
    TreeUtils.labelTree(root, "", "l");
    final StringBuilder sb = new StringBuilder();
    sb.append("\\begin{forest}\n");
    printRecTexForest(root, sb, 0, maxDepth);
    sb.append("\n").append("\\end{forest}");
    return sb.toString();
  }

  /**
   * Prints a tikz tree representing the labeled Concept Hierarchy.
   * @param node currently visited node
   * @param sb the stringbuilder that collects the String representation
   * @param depth the current depth
   * @param maxDepth the maximal depth
   */
  private static void printRecTexForest(final ConceptNode node, final StringBuilder sb, final int depth,
                                        final int maxDepth) {
    if (depth == 0) {
      sb.append("[Root\n");
    }

    if (depth < maxDepth) {
      for (ConceptNode child : node.getChildren()) {
        if (child.getCount() / (double) node.getCount() > 0.005) {
          sb.append("\t".repeat(Math.max(0, depth + 1)));
          sb.append("[").append(child.getLabel()).append(" \n");
          printRecTexForest(child, sb, depth + 1, maxDepth);
          sb.append("]");
        }
      }
    }
    if (depth == 0) {
      sb.append("]");
    }
  }

  /**
   * convenience method for printing.
   * @param roots stupid
   */
  public static void printFullTrees(final ConceptNode... roots) {
    for (final ConceptNode root : roots) {
      LOG.info("\n" + TreeUtils.printRec(root, new StringBuilder(), 0,
          TreeUtils.deepestLevel(root)));
    }
  }

  /**
   * convenience method for printing.
   * @param roots stupid
   */
  public static void printCutoffTrees(final ConceptNode... roots) {
    for (final ConceptNode root : roots) {
      LOG.info(TreeUtils.printRec(root, new StringBuilder(), 0,
          MathUtils.log2(TreeUtils.deepestLevel(root))));
    }
  }

  public static void treesToTexFile(final String dir, final ConceptNode[] nodes) throws IOException {
    treesToTexFile(dir, -1, nodes);
  }

  private static void treesToTexFile(final String prefix, final int depth, final ConceptNode... nodes)
      throws IOException {
    int i;

    int pDepth;
    for (i = 0; i < nodes.length; ++i) {
      File treeFile = getOutPath(prefix, i + "Tree.tex");
      File conceptFile = getOutPath(prefix, i + "Concepts.tex");

      if (!treeFile.exists()) {
        Files.createFile(treeFile.toPath());
      }
      if (!conceptFile.exists()) {
        Files.createFile(conceptFile.toPath());
      }

      try (FileOutputStream fos = new FileOutputStream(treeFile);
           OutputStreamWriter osw = new OutputStreamWriter(fos);
           BufferedWriter bw = new BufferedWriter(osw)) {
        if (depth == -1 ||  MathUtils.log2(TreeUtils.deepestLevel(nodes[i])) + 1 < depth) {
          pDepth = MathUtils.log2(TreeUtils.deepestLevel(nodes[i])) + 1;
        } else {
          pDepth = depth;
        }
        System.out.println("Visualization depth: " + pDepth);
        bw.write(TreeUtils.getTexForrest(nodes[i], pDepth));
        bw.flush();
      } catch (IOException e) {
        e.printStackTrace();
      }

      try (FileOutputStream fos = new FileOutputStream(conceptFile);
           OutputStreamWriter osw = new OutputStreamWriter(fos);
           BufferedWriter bw = new BufferedWriter(osw)) {
        if (depth == -1 ||  MathUtils.log2(TreeUtils.deepestLevel(nodes[i]))  + 1 < depth) {
          pDepth = MathUtils.log2(TreeUtils.deepestLevel(nodes[i]))  + 1;
        } else {
          pDepth = depth;
        }
        System.out.println("Visualization depth: " + pDepth);
        bw.write(TreeUtils.getTexTables(nodes[i], pDepth));
        bw.flush();
      } catch (IOException e) {
        e.printStackTrace();
      }

    }
  }

  private static File getOutPath(String dir, String fileName) {
    return Paths.get(Paths.get("").toString(), dir + "_" + fileName).toFile();
  }
}
