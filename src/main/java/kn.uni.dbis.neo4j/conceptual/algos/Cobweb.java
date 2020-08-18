package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of Douglas Fishers Cobweb.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 *
 */
public class Cobweb {
  private Cobweb() {
    // NOOP
  }

  /**
   * run the actual cobweb algorithm.
   *
   * @param newNode       node to incorporate
   * @param currentNode   node currently visiting
   */
  public static void cobweb(final ConceptNode newNode, final ConceptNode currentNode) {
    final float[] results = new float[4];

    final Result result = findHost(currentNode, newNode);
    final ConceptNode host = result.getNode();

    results[0] = createNewNodeCU(currentNode, newNode);
    results[1] = result.getCU();
    results[2] = mergeNodesCU(currentNode, host, result.getSecondNode());
    results[3] = splitNodesCU(host, currentNode);

    // By default keep recurring until we have a better option or reach a leaf
    float best = results[2];
    int bestIdx = 2;
    for (int i = 0; i < results.length; i++) {
      if (results[i] > best) {
        best = results[i];
        bestIdx = i;
      }
    }

    switch (bestIdx) {
      case 0 -> createNewNode(currentNode, newNode, true);
      case 1 -> {
        currentNode.updateCounts(newNode);
        cobweb(newNode, host);
      }
      case 2 -> {
        final ConceptNode mergedNode = mergeNodes(currentNode, host, result.getSecondNode(), true);
        if (mergedNode == null) {
          throw new RuntimeException("Unreachable");
        }
        currentNode.updateCounts(newNode);
        cobweb(newNode, mergedNode);
      }
      case 3 -> {
        splitNodes(host, currentNode, true);
        cobweb(newNode, currentNode);
      }
      default -> throw new RuntimeException("Invalid best operation");
    }
  }

  /**
   * finds the child most suitable to host the new node.
   *
   * @param currentNode  node currently visiting
   * @param newNode node to be hosted
   * @return the best host and the cu for that
   */
  private static Result findHost(final ConceptNode currentNode, final ConceptNode newNode) {
    float curEAPDiff;
    float maxEAPDiff = Integer.MIN_VALUE;
    int i = 0;
    int bestI = -1;
    ConceptNode clone;
    ConceptNode best = null;
    ConceptNode bestClone = null;
    ConceptNode secondBest = null;
    final ConceptNode currentNodeTemp = new ConceptNode(currentNode);
    currentNodeTemp.updateCounts(newNode);

    for (ConceptNode child : currentNode.getChildren()) {
      clone = new ConceptNode(child);
      clone.updateCounts(newNode);
      curEAPDiff = child.getCount()/ (float) currentNode.getCount()
          * (clone.getExpectedAttributePrediction() - child.getExpectedAttributePrediction());

      if (maxEAPDiff < curEAPDiff) {
        maxEAPDiff = curEAPDiff;
        secondBest = best;
        best = child;
        bestClone = clone;
        bestI = i;
      }
      i++;
    }
    if (bestI != -1) {
      currentNodeTemp.setChild(bestI, bestClone);
    }
    return new Result(computeCU(currentNodeTemp), best, secondBest);
  }

  /**
   * creates a new child for the node to incorporate and computes the cu.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   * @return the op result including the cu and the altered node
   */
  private static float createNewNodeCU(final ConceptNode currentNode, final ConceptNode newNode) {
    final ConceptNode clone = new ConceptNode(currentNode);
    if (currentNode.getId() != null) {
      final ConceptNode parentClone = new ConceptNode(currentNode.getParent());
      clone.setParent(parentClone);
    }
    clone.updateCounts(newNode);
    createNewNode(clone, newNode, false);
    return (currentNode.getId() != null) ? computeCU(clone.getParent()) : computeCU(clone);
  }

  /**
   * creates a new child for the node to incorporate.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   * @param setParent flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                  alter the tree when probing an action)
   */
  private static void createNewNode(final ConceptNode currentNode, final ConceptNode newNode, final boolean setParent) {
    if (currentNode.getId() != null) {
      final ConceptNode parent = currentNode.getParent();
      // we are in a leaf node. leaf nodes are concrete data instances and shall stay leaves
      // remove the leaf from it's current parent
      parent.removeChild(currentNode);
      // make a new node containing the same count and attributes
      final ConceptNode conceptNode = new ConceptNode(currentNode);
      // add the new node as intermediate between the current leaf and its parent
      parent.addChild(conceptNode);
      // set the intermediate nodes id to null as its an inner node
      conceptNode.setId(null);
      // update the attribute counts to incorporate the new node
      conceptNode.updateCounts(newNode);
      // add the leaf and the new node as children
      conceptNode.addChild(newNode);
      conceptNode.addChild(currentNode);
      currentNode.setParent(conceptNode);
      if (setParent) {
        newNode.setParent(conceptNode);
      }
    } else {
      currentNode.updateCounts(newNode);
      currentNode.addChild(newNode);
      if (setParent) {
        newNode.setParent(currentNode);
      }
    }
  }

  /**
   * clones the current node, splits & computes the cu.
   *
   * @param host    node to be split
   * @param current node to append children of host
   * @return op result including altered node and the cu
   */
  private static float splitNodesCU(final ConceptNode host, final ConceptNode current) {
    if (host == null || host.getChildren().size() == 0) {
      return Integer.MIN_VALUE;
    }
    final ConceptNode clone = new ConceptNode(current);
    splitNodes(host, clone, false);
    return computeCU(clone);
  }

  /**
   * splits the host node and appends its children to the current node.
   *
   * @param host    node to be split
   * @param current node to append children of host
   * @param setParent flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                 alter the tree when probing an action)
   */
  private static void splitNodes(final ConceptNode host, final ConceptNode current, final boolean setParent) {
    if (host == null) {
      return;
    }
    for (ConceptNode child : host.getChildren()) {
      if (setParent) {
        child.setParent(current);
      }
      current.addChild(child);
    }
    current.removeChild(host);
    if (setParent) {
      host.setParent(null);
      host.getChildren().clear();
    }
  }

  /**
   * clones the actual current node, merges and computes the cu.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @return op result including altered node and the cu
   */
  private static float mergeNodesCU(final ConceptNode current, final ConceptNode host, final ConceptNode secondHost) {
    final ConceptNode clonedParent = new ConceptNode(current);
    float cu = (mergeNodes(clonedParent, host, secondHost, false) != null) ? computeCU(clonedParent)
        : Integer.MIN_VALUE;
    return cu == 0.0f ? -1.0f : cu;
  }

  /**
   * merges the host node with the next best host and appends its children to the resulting node.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @param setParent flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                  alter the tree when probing an action)
   * @return the node that results by the merge.
   */
  private static ConceptNode mergeNodes(final ConceptNode current, final ConceptNode host, final ConceptNode secondHost,
                                 final boolean setParent) {
    if (host == null || secondHost == null) {
      return null;
    }
    current.removeChild(host);
    current.removeChild(secondHost);
    if (setParent) {
      secondHost.setParent(null);
      host.setParent(null);
    }

    final ConceptNode mNode = new ConceptNode(host);
    mNode.clearChildren();
    mNode.setId(null);
    mNode.updateCounts(secondHost);
    mNode.addChild(host);
    mNode.addChild(secondHost);
    if (setParent) {
      host.setParent(mNode);
      secondHost.setParent(mNode);
      mNode.setParent(current);
    }
    current.addChild(mNode);

    return mNode;
  }

  /**
   * computes the category utility as defined by fisher 1987.
   *
   * @param parent    the node which acts as reference node wrt. the children/partitions of the concepts
   * @return the category utility
   */
  private static float computeCU(final ConceptNode parent) {
    final float parentChildCount = parent.getChildren().size();
    if (parentChildCount == 0) {
      return Integer.MIN_VALUE;
    }
    final float parentEAP = parent.getExpectedAttributePrediction();
    float cu = 0.0f;
    final float parentCount = parent.getCount();
    for (ConceptNode child : parent.getChildren()) {
      cu += ((float) child.getCount()) / parentCount
          * (child.getExpectedAttributePrediction() - parentEAP);
    }
    return cu / parentChildCount;
  }

  /**
   * Used for passing the pair cu, node of the find hosts methods back to the calling method.
   */
  private static final class Result {
    /**
     * the cu that the operation yields.
     */
    private final float cu;
    /**
     * the node with the operation applied.
     */
    private final ConceptNode node1;
    /**
     * the node with the operation applied.
     */
    private final ConceptNode node2;

    /**
     * Constructor.
     *
     * @param cu        the cu that the operation yields.
     * @param node1      the node that yield the best eap
     * @param node2      the node that yield the second best eap
     */
    private Result(final float cu, final ConceptNode node1, final ConceptNode node2) {
      this.cu = cu;
      this.node1 = node1;
      this.node2 = node2;
    }

    /**
     * getter.
     *
     * @return the cu
     */
    float getCU() {
      return this.cu;
    }

    /**
     * getter.
     *
     * @return the concept node
     */
    ConceptNode getNode() {
      return this.node1;
    }

    /**
     * getter.
     *
     * @return the concept node
     */
    ConceptNode getSecondNode() {
      return this.node2;
    }
  }

  /**
   * Hungarian algorithm to implement what is described in literature/conceptual_clust/trestle.pdf
   *
   * matches a nodes attribute name to the most fitting ones of the root.
   * Uses the category utility to determine how good it fits.
   *
   * @param root the root node of the current concept tree
   * @param toMatch the concept node to be matched against the root node
  */
  public void match(final ConceptNode root, final ConceptNode toMatch) {
    final List<String> rootAttribs = new ArrayList<>(root.getAttributes().keySet());
    final List<String> toMatchAttrib = new ArrayList<>(toMatch.getAttributes().keySet());

    final float baseEAP = toMatch.getExpectedAttributePrediction();
    final float[][] costMatrix = new float[toMatchAttrib.size()][rootAttribs.size()];
    ConceptNode altered;
    int i = 0;
    int j = 0;
    float min;
    final int[] minIdx = new int[toMatchAttrib.size()];
    for (String toMatchName : toMatchAttrib) {
      min = 1;
      for (String rootName : rootAttribs) {
        altered = new ConceptNode(toMatch);
        altered.getAttributes().put(rootName, altered.getAttributes().get(toMatchName));
        altered.getAttributes().remove(toMatchName);
        costMatrix[j][i] = 1 - (altered.getExpectedAttributePrediction() - baseEAP);
        if (costMatrix[j][i] < min) {
          min = costMatrix[j][i];
          minIdx[j] = i;
        }
        i++;
      }
      j++;
    }
    // Transform node
    for (j = 0; j < costMatrix.length; j++) {
      if (costMatrix[j][minIdx[j]] < 1) {
        toMatch.getAttributes()
            .put(rootAttribs.get(minIdx[j]), toMatch.getAttributes().get(toMatchAttrib.get(j)));
        toMatch.getAttributes().remove(toMatchAttrib.get(j));
      }
    }
  }
}
