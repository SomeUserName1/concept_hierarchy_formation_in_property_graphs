package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.List;
import java.util.Map;

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
    final double[] results = new double[4];

    final Result result = findHost(currentNode, newNode);
    results[0] = result.getCu();
    final ConceptNode host = result.getNode();

    results[1] = createNewNodeCU(currentNode, newNode);
    results[2] = mergeNodesCU(currentNode, host, newNode);
    results[3] = splitNodesCU(host, newNode);

    // By default take create new as standard action if no other is better
    double best = results[1];
    int bestIdx = 1;
    for (int i = 0; i < results.length; i++) {
      if (results[i] > best) {
        best = results[i];
        bestIdx = i;
      }
    }

    switch (bestIdx) {
      case 1:
        createNewNode(currentNode, newNode, true);
        break;
      case 2:
        final ConceptNode mergedNode = mergeNodes(currentNode, host, newNode, true);
        if (mergedNode == null) {
          throw new RuntimeException("Unreachable");
        }
        currentNode.updateCounts(newNode);
        cobweb(newNode, mergedNode);
        break;
      case 3:
        splitNodes(host, currentNode, true);
        cobweb(newNode, currentNode);
        break;
      case 0:
        currentNode.updateCounts(newNode);
        cobweb(newNode, host);
        break;
      default:
        throw new RuntimeException("Invalid best operation");
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
    double curCU;
    double maxCU = Integer.MIN_VALUE;
    int i = 0;
    ConceptNode clone;
    ConceptNode best = null;
    final ConceptNode currentNodeTemp = new ConceptNode(currentNode);
    currentNodeTemp.updateCounts(newNode);
    ConceptNode currentNodeClone;
    final double currentNodeEAP = getExpectedAttributePrediction(currentNode);

    for (ConceptNode child : currentNode.getChildren()) {
      clone = new ConceptNode(child);
      clone.updateCounts(newNode);

      currentNodeClone = new ConceptNode(currentNodeTemp);
      currentNodeClone.setChild(i, clone);

      curCU = computeCU(currentNodeClone, currentNodeEAP);
      if (maxCU < curCU) {
        maxCU = curCU;
        best = child;
      }
      i++;
    }
    return new Result(maxCU, best);
  }

  /**
   * creates a new child for the node to incorporate and computes the cu.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   * @return the op result including the cu and the altered node
   */
  private static double createNewNodeCU(final ConceptNode currentNode, final ConceptNode newNode) {
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
  private static double splitNodesCU(final ConceptNode host, final ConceptNode current) {
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
    }
  }

  /**
   * clones the actual current node, merges and computes the cu.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @param newNode node to be incorporated
   * @return op result including altered node and the cu
   */
  private static double mergeNodesCU(final ConceptNode current, final ConceptNode host, final ConceptNode newNode) {
    final ConceptNode clonedParent = new ConceptNode(current);
    return (mergeNodes(clonedParent, host, newNode, false) != null) ? computeCU(clonedParent)
        : Integer.MIN_VALUE;
  }

  /**
   * merges the host node with the next best host and appends its children to the resulting node.
   *
   * @param host    node to be merged
   * @param current parent of the host
   * @param newNode node to be incorporated
   * @param setParent flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                  alter the tree when probing an action)
   * @return the node that results by the merge.
   */
  private static ConceptNode mergeNodes(final ConceptNode current, final ConceptNode host, final ConceptNode newNode,
                                 final boolean setParent) {
    if (host == null) {
      return null;
    }
    current.getChildren().remove(host);

    final Result secondHost = findHost(current, newNode);
    if (secondHost.node == null) {
      return null;
    }
    current.removeChild(secondHost.node);
    if (setParent) {
      secondHost.node.setParent(null);
      host.setParent(null);
    }

    final ConceptNode mNode = new ConceptNode(host);
    mNode.clearChildren();
    mNode.setId(null);
    mNode.updateCounts(secondHost.node);
    mNode.addChild(host);
    mNode.addChild(secondHost.node);
    if (setParent) {
      host.setParent(mNode);
      secondHost.node.setParent(mNode);
      mNode.setParent(current);
    }
    current.addChild(mNode);

    return mNode;
  }

  /**
   * computes the category utility as defined by fisher 1987.
   *
   * @param parent the node which acts as reference node wrt. the children/partitions of the concepts
   * @return the category utility
   */
  private static double computeCU(final ConceptNode parent) {
    if (parent.getChildren().size() == 0) {
      return Integer.MIN_VALUE;
    }
    final double parentEAP = getExpectedAttributePrediction(parent);
    return computeCU(parent, parentEAP);
  }

  /**
   * computes the category utility as defined by fisher 1987.
   *
   * @param parent    the node which acts as reference node wrt. the children/partitions of the concepts
   * @param parentEAP precomputed Expected Attribute Prediction Probability for the parent node to avoid recomputation
   * @return the category utility
   */
  private static double computeCU(final ConceptNode parent, final double parentEAP) {
    final double parentChildCount = parent.getChildren().size();
    if (parentChildCount == 0) {
      return Integer.MIN_VALUE;
    }
    double cu = 0.0;
    final double parentCount = parent.getCount();
    for (ConceptNode child : parent.getChildren()) {
      cu += (double) child.getCount() / parentCount
          * (getExpectedAttributePrediction(child) - parentEAP);
    }
    return cu / parentChildCount;
  }

  /**
   * Computes the Expected Attribute Prediction Probability for the current concept node.
   *
   * @param category the node for which to compute the cu for.
   * @return the EAP
   */
  private static double getExpectedAttributePrediction(final ConceptNode category) {
    final double noAttributes = category.getAttributes().size();
    if (noAttributes == 0) {
      return 0;
    }
    double exp = 0;
    final double total = category.getCount();
    double intermediate;
    NumericValue num;

    for (Map.Entry<String, List<Value>> attrib : category.getAttributes().entrySet()) {
      for (Value val : attrib.getValue()) {
        if (val instanceof NumericValue) {
          num = (NumericValue) val;
          exp += 1.0 / (num.getStd()/num.getMean() + 1) - 1;
        } else {
          intermediate = (double) val.getCount() / total;
          exp += intermediate * intermediate;
        }
      }
    }
    return exp / noAttributes;
  }

  /**
   * Used for passing the pair cu, node of the find hosts methods back to the calling method.
   */
  private static final class Result {
    /**
     * the cu that the operation yields.
     */
    private final double cu;
    /**
     * the node with the operation applied.
     */
    private final ConceptNode node;

    /**
     * Constructor.
     *
     * @param cu        the cu that the operation yields.
     * @param node      the node with the operation applied.
     */
    private Result(final double cu, final ConceptNode node) {
      this.cu = cu;
      this.node = node;
    }

    /**
     * getter.
     *
     * @return the cu
     */
    double getCu() {
      return this.cu;
    }

    /**
     * getter.
     *
     * @return the concept node
     */
    ConceptNode getNode() {
      return this.node;
    }
  }

  /*
   * matches a nodes attribute name to the most fitting ones of the root.
   * Uses the category utility to determine how good it fits.
   *
   * @param toMatch the concept node to be matched

  public void match(final ConceptNode toMatch) {
    final List<String> rootAttribs = new ArrayList<>(this.root.getAttributes().keySet());
    final List<String> toMatchAttrib = new ArrayList<>(toMatch.getAttributes().keySet());

    final double baseEAP = this.getExpectedAttributePrediction(toMatch);
    final double[][] costMatrix = new double[toMatchAttrib.size()][rootAttribs.size()];
    ConceptNode altered;
    int i = 0;
    int j = 0;
    double min;
    final int[] minIdx = new int[toMatchAttrib.size()];
    for (String toMatchName : toMatchAttrib) {
      min = 1;
      for (String rootName : rootAttribs) {
        altered = new ConceptNode(toMatch);
        altered.getAttributes().put(rootName, altered.getAttributes().get(toMatchName));
        altered.getAttributes().remove(toMatchName);
        costMatrix[j][i] = 1 - (this.getExpectedAttributePrediction(altered) - baseEAP);
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
  }*/
}
