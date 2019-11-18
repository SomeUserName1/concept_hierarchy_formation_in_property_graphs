package kn.uni.dbis.neo4j.conceptual.util;

import kn.uni.dbis.neo4j.conceptual.algos.ConceptNode;

/**
 * Utilities for trees.
 *  @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class TreeUtils {
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
}
