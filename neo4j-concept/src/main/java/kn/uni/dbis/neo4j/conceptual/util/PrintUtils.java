package kn.uni.dbis.neo4j.conceptual.util;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.Value;

/**
 * PrintUtils.
 *  @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class PrintUtils {
  /** Logger. */
  private static final Logger LOG = Logger.getLogger("PropertyGraphCobweb");

  /**
   * Hidden default constructor.
   */
  private PrintUtils() {
    // NOOP
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
  private static String getTexTree(final ConceptNode root, final int maxDepth) {
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
  public void printFullTrees(final ConceptNode... roots) {
    for (final ConceptNode root : roots) {
      LOG.info(PrintUtils.printRec(root, new StringBuilder(), 0,
          TreeUtils.deepestLevel(root)));
    }
  }

  /**
   * convenience method for printing.
   * @param roots stupid
   */
  public static void printCutoffTrees(final ConceptNode... roots) {
    for (final ConceptNode root : roots) {
      LOG.info(PrintUtils.printRec(root, new StringBuilder(), 0,
          MathUtils.log2(TreeUtils.deepestLevel(root))));
    }
  }

  /**
   * convenience method for printing.
   * @param root stupid
   */
  public static void prettyPrint(final ConceptNode root) {
    final int cut = MathUtils.log2(TreeUtils.deepestLevel(root));
    LOG.info(PrintUtils.printRec(root, new StringBuilder(), 0, cut));
    LOG.info(getTexTree(root, cut));
  }
}
