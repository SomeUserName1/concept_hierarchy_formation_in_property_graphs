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
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

/**
 * Utilities for trees.
 *  @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class TreeUtils {
  /** Logger. */
  public static final Logger LOG = Logger.getLogger("PropertyGraphCobweb");
  
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
      return 0;
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
      for (int i = 0; i < depth; i++) {
        sb.append("\t");
      }
      sb.append("|____");
    }

    sb.append(node.toString()).append("\n");
    if (depth <= maxDepth) {
      final List<ConceptNode> localChildren;
      synchronized (node.getChildren()) {
        localChildren = new ArrayList<>(node.getChildren());
      }
      for (ConceptNode child : localChildren) {
        printRec(child, sb, depth + 1, maxDepth);
      }
    }
    return sb.toString();
  }

  /**
   * Returns a Latex table representation of the ConceptNode.
   * @param node stupid
   * @return a String containing a latex table representation of the ConceptNode
   */
  public String toTexTable(final ConceptNode node) {
    final StringBuilder sb = new StringBuilder();
    sb.append("ConceptNode \\hspace{1cm} P(node) = ")
        .append(Double.toString((double) node.getCount() / (double) node.getParent().getCount()), 0, 5)
        .append("\\\\ Attributes: \\\\ \\begin{tabular}{|c|c|c|} \\hline");
    for (ConcurrentMap.Entry<String, List<Value>> attribute : node.getAttributes().entrySet()) {
      sb.append("\\multirow{4}{*}{").append(attribute.getKey()).append("} ");
      synchronized (attribute.getValue()) {
        for (Value value : attribute.getValue()) {
          sb.append(value.toTexString()).append((double) value.getCount() / (double) node.getCount())
              .append("\\\\ \\hline");
        }
      }
    }
    sb.append("\\end{tabular}");
    return sb.toString();
  }

  /**
   * Prints a tikz tree representing the labeled Concept Hierarchy.
   * @param node currently visited node
   * @param sb the stringbuilder that collects the String representation
   * @param depth the current depth
   * @param maxDepth the maximal depth
   */
  private static void printRecTexTree(final ConceptNode node, final StringBuilder sb, final int depth,
                                      final int maxDepth) {
    if (depth == 0) {
      sb.append("\\node {Root}\n");
    }

    if (depth <= maxDepth) {
      for (ConceptNode child : node.getChildren()) {
        for (int i = 0; i <= depth; i++) {
          sb.append("\t");
        }
        sb.append("child { node {").append(child.getLabel()).append("} ");
        if (child.getChildren().size() > 0) {
          sb.append("\n");
          printRecTexTree(child, sb, depth + 1, maxDepth);
        }
        sb.append("}");
      }
    }
  }

  /**
   * Constructs a tikzpicture containing a tree representation of the concept hierarchy.
   * @param root the root of the tree to be visualized
   * @param maxDepth the maximal depth to be visualized
   * @return a String containing a tkizpicture
   */
  public static String getTexTree(final ConceptNode root, final int maxDepth) {
    TreeUtils.labelTree(root, "", "l");
    final StringBuilder sb = new StringBuilder();
    sb.append("\\begin{tikzpicture}[sibling distance=10em, "
        + "every node/.style = {shape=rectangle, rounded corners, "
        + "draw, align=center,"
        + "top color=white, bottom color=blue!20}]]");
    printRecTexTree(root, sb, 0, maxDepth);
    sb.append(";\n").append("\\end{tikzpicture}");
    return sb.toString();
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

  /**
   * convenience method for printing.
   * @param root stupid
   */
  public static void prettyPrint(final ConceptNode root) {
    final int cut = MathUtils.log2(TreeUtils.deepestLevel(root));
    LOG.info(TreeUtils.printRec(root, new StringBuilder(), 0, cut));
    LOG.info(getTexTree(root, cut));
  }

  public static void treesToTexFile(ConceptNode[] nodes, String dir) throws IOException {
    if (nodes.length != 4) {
      throw new RuntimeException("Need exactly 4 Trees");
    }

    final File[] files = {getOutPath(dir, "NodePropertiesConcepts.tex"),
        getOutPath(dir, "RelationPropertiesConcepts.tex"),
        getOutPath(dir, "NodeStructuralFeaturesConcepts.tex"),
        getOutPath(dir, "NodeSummaryConcepts.tex")};

    if (!files[0].getParentFile().exists()) {
      Files.createDirectory(files[0].getParentFile().toPath());
    }

    for (int i = 0; i < nodes.length; ++i) {
      if (!files[i].exists()) {
        Files.createFile(files[i].toPath());
      }
      try (FileOutputStream fos = new FileOutputStream(files[i]);
           OutputStreamWriter osw = new OutputStreamWriter(fos);
           BufferedWriter bw = new BufferedWriter(osw)) {
        bw.write(TreeUtils.getTexTree(nodes[i], MathUtils.log2(TreeUtils.deepestLevel(nodes[3]))));
        bw.flush();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private static File getOutPath(String dir, String fileName) {
    return Paths.get(Paths.get("").toString(), dir, fileName).toFile();
  }
}
