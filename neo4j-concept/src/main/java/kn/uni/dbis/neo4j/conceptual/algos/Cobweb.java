package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.logging.Logger;

import javax.annotation.concurrent.ThreadSafe;

import com.google.common.util.concurrent.AtomicDouble;

import kn.uni.dbis.neo4j.conceptual.util.ResourceLock;

import static kn.uni.dbis.neo4j.conceptual.util.LockUtils.lockAll;

/**
 * Implementation of Douglas Fishers Cobweb.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@ThreadSafe
public final class Cobweb {
  /**
   * Logger.
   */
  static final Logger LOG = Logger.getLogger("PropertyGraphCobweb");

  /**
   * hidden default constructor.
   */
  private Cobweb() {
    // NOOP
  }
/*

   * @param threadPool  used to execute probing
  ,
                            final ExecutorService threadPool
      try {
    final Future<Double> createFuture = threadPool.submit(() -> createNewNodeCU(currentNode, newNode));
    final Result findResult = threadPool.submit(() ->findHost(currentNode, newNode)).get();
    final ConceptNode host = findResult.getNode();
    final Future<Double> mergeFuture = threadPool.submit(() -> mergeNodesCU(currentNode, host, newNode));
    final Future<Double> splitFuture = threadPool.submit(() -> splitNodesCU(host, newNode));

    final double[] results = {findResult.getCu(), createFuture.get(), mergeFuture.get(), splitFuture.get()};


    // By default take create new as standard action if no other is better
    double best = results[1];
    int bestIdx = 1;
    for (int i = 0; i < results.length; i++) {
      if (results[i] > best) {
        best = results[i];
        bestIdx = i;
      }
    }
*/


  /**
   * run the actual cobweb algorithm.
   *
   * @param newNode     node to incorporate
   * @param currentNode node currently visiting
   */
  public static void cobweb(final ConceptNode newNode, final ConceptNode currentNode) {

    final Result findResult = findHost(currentNode, newNode);
    final ConceptNode host = findResult.getNode();
    final double[] results = {findResult.getCu(), createNewNodeCU(currentNode, newNode),
        mergeNodesCU(currentNode, host, newNode), splitNodesCU(host, newNode)};

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
        try (ResourceLock ignored = lockAll(currentNode.getLock().writeLock(), newNode.getLock().readLock())) {
          currentNode.updateCounts(newNode);
        }
        cobweb(newNode, mergedNode);
        break;
      case 3:
        splitNodes(host, currentNode, true);
        cobweb(newNode, currentNode);
        break;
      case 0:
        try (ResourceLock ignored = lockAll(currentNode.getLock().writeLock(), currentNode.getLock().readLock())) {
          currentNode.updateCounts(newNode);
        }
        cobweb(newNode, host);
        break;
      default:
        throw new RuntimeException("Invalid best operation");
    }
  }

  /**
   * finds the child most suitable to host the new node.
   *
   * @param currentNode node currently visiting
   * @param newNode     node to be hosted
   * @return the best host and the cu for that
   */
  private static Result findHost(final ConceptNode currentNode, final ConceptNode newNode) {
    double curCU;
    double maxCU = Integer.MIN_VALUE;
    int count = 0;
    ConceptNode clone;
    ConceptNode best = null;

    ArrayList<Lock> locks = new ArrayList<>();
    for (ConceptNode child : currentNode.getChildren()) {
      locks.add(child.getLock().readLock());
    }
    locks.add(currentNode.getLock().readLock());
    locks.add(newNode.getLock().readLock());

    try (ResourceLock ignored = lockAll(locks)) {
      final ConceptNode currentNodeTemp = new ConceptNode(currentNode);
      currentNodeTemp.updateCounts(newNode);
      ConceptNode currentNodeClone;
      final double currentNodeEAP = getExpectedAttributePrediction(currentNode);

      for (final ConceptNode child : currentNode.getChildren()) {
        clone = new ConceptNode(child);
        clone.updateCounts(newNode);

        currentNodeClone = new ConceptNode(currentNodeTemp);

        currentNodeClone.setChild(count, clone);

        curCU = computeCU(currentNodeClone, currentNodeEAP);
        if (maxCU < curCU) {
          maxCU = curCU;
          best = child;
        }
        count++;
      }

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
    final double cu;
    try (ResourceLock ignored = lockAll(currentNode.getLock().readLock(),
        currentNode.getParent().getLock().readLock())) {

      final ConceptNode clone = new ConceptNode(currentNode);
      if (currentNode.getId() != null) {
        final ConceptNode parentClone = new ConceptNode(currentNode.getParent());
        clone.setParent(parentClone);
      }
      clone.updateCounts(newNode);
      createNewNode(clone, newNode, false);
      cu = currentNode.getId() == null ? computeCU(clone) : computeCU(clone.getParent());
    }
    return cu;
  }

  /**
   * creates a new child for the node to incorporate.
   *
   * @param currentNode not to add the child to
   * @param newNode     to to be added
   * @param setParent   flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                    alter the tree when probing an action)
   */
  private static void createNewNode(final ConceptNode currentNode, final ConceptNode newNode,
                                    final boolean setParent) {
    final int holdCount = currentNode.getParent().getLock().getReadHoldCount();
    if (!setParent) {
      for (int i = 0; i < holdCount; i++) {
        currentNode.getParent().getLock().readLock().unlock();
      }
    }
    // FIXME create fails; CU may be /is wrong
    try (ResourceLock ignored = lockAll(currentNode.getParent().getLock().writeLock(),
        currentNode.getLock().writeLock(), newNode.getLock().writeLock())) {
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
    if (!setParent) {
      for (int i = 0; i < holdCount; i++) {
        currentNode.getParent().getLock().readLock().lock();
      }
    }

  }

  /**
   * clones the current node, splits & computes the cu.
   *
   * @param host        node to be split
   * @param currentNode node to append children of host
   * @return op result including altered node and the cu
   */
  private static double splitNodesCU(final ConceptNode host, final ConceptNode currentNode) {
    if (host == null) {
      return Integer.MIN_VALUE;
    }
    final double cu;
    try (ResourceLock ignored = lockAll(currentNode.getLock().readLock(), host.getLock().readLock())) {
      if (host.getChildren().size() == 0) {
        return Integer.MIN_VALUE;
      }
      final ConceptNode clone = new ConceptNode(currentNode);
      splitNodes(host, clone, false);
      cu = computeCU(clone);
    }
    return cu;
  }

  /**
   * splits the host node and appends its children to the current node.
   *
   * @param host      node to be split
   * @param current   node to append children of host
   * @param setParent flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                  alter the tree when probing an action)
   */
  private static void splitNodes(final ConceptNode host, final ConceptNode current,
                                 final boolean setParent) {
    if (host == null) {
      return;
    }

    ArrayList<Lock> locks = new ArrayList<>();
    for (ConceptNode child : host.getChildren()) {
      if (setParent) {
        locks.add(child.getLock().writeLock());
      } else {
        locks.add(child.getLock().readLock());
      }
    }
    locks.add(current.getLock().writeLock());
    locks.add(host.getLock().readLock());

    try (ResourceLock ignored = lockAll(locks)) {
      for (final ConceptNode child : host.getChildren()) {
        if (setParent) {
          child.setParent(current);
          current.addChild(child);
        } else {
          current.addChild(child);
        }
      }
      current.removeChild(host);
    }
    if (setParent) {
      try (ResourceLock ignored1 = lockAll(host.getLock().writeLock())) {
        host.setParent(null);
      }
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
  private static double mergeNodesCU(final ConceptNode current, final ConceptNode host,
                                     final ConceptNode newNode) {
    final double cu;
    try (ResourceLock ignored = lockAll(current.getLock().readLock())) {
      if (current.getChildren().size() < 2) {
        return Integer.MIN_VALUE;
      }
      final ConceptNode clonedParent = new ConceptNode(current);
      cu = (mergeNodes(clonedParent, host, newNode, false) != null) ? computeCU(clonedParent)
          : Integer.MIN_VALUE;
    }
    return cu;
  }

  /**
   * merges the host node with the next best host and appends its children to the resulting node.
   *
   * @param host      node to be merged
   * @param current   parent of the host
   * @param newNode   node to be incorporated
   * @param setParent flag indicating weather to set the parent pointers (false when computing the cu in order not to
   *                  alter the tree when probing an action)
   * @return the node that results by the merge.
   */
  private static ConceptNode mergeNodes(final ConceptNode current, final ConceptNode host,
                                        final ConceptNode newNode, final boolean setParent) {
    if (host == null) {
      return null;
    }
    final ConceptNode secondHost;
    final ConceptNode mNode;

    ArrayList<Lock> locks = new ArrayList<>();
    locks.add(current.getLock().writeLock());
    if (setParent) {
      locks.add(host.getLock().writeLock());
    } else {
      locks.add(host.getLock().readLock());
    }

    try (ResourceLock ignored1 = lockAll(locks)) {

      current.removeChild(host);

      secondHost = findHost(current, newNode).getNode();
      if (secondHost == null) {
        return null;
      }
      locks.clear();
      if (setParent) {
        locks.add(secondHost.getLock().writeLock());
      } else {
        locks.add(secondHost.getLock().readLock());
      }
      try (ResourceLock ignored2 = lockAll(locks)) {
        current.removeChild(secondHost);

        mNode = new ConceptNode(host);
        mNode.clearChildren();
        mNode.setId(null);
        mNode.updateCounts(secondHost);
        mNode.addChild(host);
        mNode.addChild(secondHost);

        if (setParent) {
          host.setParent(mNode);
          secondHost.setParent(mNode);
          mNode.setParent(current);
          current.addChild(mNode);
        }
      }
    }

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
      return 0;
    }
    final double cu;
    try (ResourceLock ignored = lockAll(parent.getLock().readLock())) {
      final double parentEAP = getExpectedAttributePrediction(parent);
      cu = computeCU(parent, parentEAP);
    }
    return cu;
  }

  /**
   * computes the category utility as defined by fisher 1987.
   *
   * @param parent    the node which acts as reference node wrt. the children/partitions of the concepts
   * @param parentEAP precomputed Expected Attribute Prediction Probability for the parent node to avoid recomputation
   * @return the category utility
   */
  private static double computeCU(final ConceptNode parent, final double parentEAP) {
    double cu = 0.0;

    ArrayList<Lock> locks = new ArrayList<>();
    for (ConceptNode child : parent.getChildren()) {
      locks.add(child.getLock().readLock());
    }
    locks.add(parent.getLock().readLock());

    try (ResourceLock ignored = lockAll(locks)) {
      final double parentChildCount = parent.getChildren().size();
      if (parentChildCount == 0) {
        return Integer.MIN_VALUE;
      }

      final double parentCount = parent.getCount();
      for (final ConceptNode child : parent.getChildren()) {
          cu += (double) child.getCount() / parentCount
              * (getExpectedAttributePrediction(child) - parentEAP);
        }
        cu = cu / parentChildCount;
    }
    return cu;
  }

  /**
   * Computes the Expected Attribute Prediction Probability for the current concept node.
   *
   * @param category the node for which to compute the cu for.
   * @return the EAP
   */
  private static double getExpectedAttributePrediction(final ConceptNode category) {
    final double eap;
    try (ResourceLock ignored = lockAll(category.getLock().readLock())) {
      final double noAttributes = category.getAttributes().size();
      if (noAttributes == 0) {
        return 0;
      }

      double exp = 0;
      final double total = category.getCount();
      double intermediate;
      NumericValue num;

      for (final Map.Entry<String, List<Value>> attrib : category.getAttributes().entrySet()) {
        for (final Value val : attrib.getValue()) {
          try (ResourceLock ignored1 = lockAll(val.getLock().readLock())) {
            if (val instanceof NumericValue) {
              num = (NumericValue) val;
              exp += 1.0 / (num.getStd() / num.getMean() + 1) - 1;
            } else {
              intermediate = (double) val.getCount() / total;
              exp += intermediate * intermediate;
            }
          }
        }
      }
      eap = exp / noAttributes;
    }
    return eap;
  }

  /**
   * Used for passing the pair cu, node of the find hosts methods back to the calling method.
   */
  private static final class Result {
    /**
     * the cu that the operation yields.
     */
    private final AtomicDouble cu = new AtomicDouble(0);
    /**
     * the node with the operation applied.
     */
    private final AtomicReference<ConceptNode> node = new AtomicReference<>(null);

    /**
     * Constructor.
     *
     * @param cu   the cu that the operation yields.
     * @param node the node with the operation applied.
     */
    Result(final double cu, final ConceptNode node) {
      this.cu.set(cu);
      this.node.set(node);
    }

    /**
     * getter.
     *
     * @return the cu
     */
    double getCu() {
      return this.cu.get();
    }

    /**
     * getter.
     *
     * @return the concept node
     */
    ConceptNode getNode() {
      return this.node.get();
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
