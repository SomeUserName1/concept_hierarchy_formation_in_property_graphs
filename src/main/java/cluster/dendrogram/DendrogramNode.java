package cluster.dendrogram;

import cluster.DistanceFunction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import preprocess.DataObject;


// TODO: Tests

/**
 * This class represents a node in a Dendrogram.
 *
 * @param <T> DataObject to be wrapped in the dendrogram node. E.g. a
 *            YelpBusiness
 */
public class DendrogramNode<T extends DataObject> {
  /**
   * previously merged node to form the current.
   */
  private DendrogramNode<T> leftChild;
  /**
   * previously merged node to form the current.
   */
  private DendrogramNode<T> rightChild;
  /**
   * List containing the concrete data instances per cluster.
   */
  private List<T> cluster = new ArrayList<>();
  /**
   * distance vector from the current node to all others in the workspace of
   * the dendrogram.
   */
  private Map<DendrogramNode<T>, Integer> distances = new HashMap<>();
  /**
   * current minimal distance.
   */
  private int minimumDistance = Integer.MAX_VALUE;
  /**
   * node having the current minimal distance.
   */
  private DendrogramNode<T> minNode = null;
  /**
   * flag to indicate wether the node is a leaf, thus has no children.
   */
  private boolean isLeaf;

  DendrogramNode(DendrogramNode<T> left, DendrogramNode<T> right) {
    this.isLeaf = false;

    this.leftChild = left;
    this.rightChild = right;

    this.cluster.addAll(this.leftChild.getCluster());
    this.cluster.addAll(this.rightChild.getCluster());
  }

  DendrogramNode(T b) {
    this.isLeaf = true;

    this.leftChild = null;
    this.rightChild = null;

    this.cluster.add(b);
  }

  /**
   * used for visualization.
   *
   * @return The left node that was merged to form this node
   */
  public DendrogramNode<T> getLeftChild() {
    if (this.isLeaf) {
      throw new NullPointerException("A Leaf node has no children!");
    }
    return this.leftChild;
  }

  /**
   * used for visualization.
   *
   * @return The left node that was merged to form this node
   */
  public DendrogramNode<T> getRightChild() {
    if (this.isLeaf) {
      throw new NullPointerException("A Leaf node has no children!");
    }
    return this.rightChild;
  }


  List<T> getCluster() {
    return this.cluster;
  }

  /**
   * calls the specified distance function, adds the result to the current
   * nodes distance vector, sets it in the other node too (as linkage
   * distances are symmetric we dont need to compute it twice) and updates
   * the minimum distance and the corresponding node if it's smaller than the
   * previous minimum.
   *
   * @param node             the node for which to calculate the distance to from the
   *                         current
   * @param distanceFunction the distance function to use (see
   *                         LinkageDistance.java)
   */
  void calculateDistance(DendrogramNode<T> node,
                         DistanceFunction<T> distanceFunction) {
    int distance = distanceFunction.calculate(this, node);
    this.distances.put(node, distance);
    node.setDistance(this, distance);
    if (distance < this.minimumDistance) {
      this.minimumDistance = distance;
      this.minNode = node;
    }
  }


  Integer getDistance(DendrogramNode node) {
    return this.distances.get(node);
  }

  /**
   * if two nodes were merged, the distances to them remain in the
   * current-node local distance vector. This method drops the entry for
   * the corresponding node and either sets a new minimum if there are nodes
   * in the distance vector left or sets it to null if the removed was the
   * last node in the distance vector.
   *
   * @param node to drop from the distance vector.
   */
  void dropDistance(DendrogramNode<T> node) {
    this.distances.remove(node);
    if (node == this.minNode) {
      if (!this.distances.isEmpty()) {
        this.minimumDistance = Collections.min(this.distances.values());
        this.minNode =
            this.distances.entrySet().stream()
                .filter(entry -> entry.getValue() == this.minimumDistance)
                .map(Map.Entry::getKey).findFirst()
                .orElseThrow(NullPointerException::new);
      } else {
        this.minimumDistance = Integer.MAX_VALUE;
        this.minNode = null;
      }
    }
  }

  private void setDistance(DendrogramNode<T> node, Integer distance) {
    this.distances.put(node, distance);
    if (distance < this.minimumDistance) {
      this.minimumDistance = distance;
      this.minNode = node;
    }
  }

  // Geklaut vom B-Tree assignment in dbai
  public String toNString(int level) {
    final StringBuilder sb = new StringBuilder();
    for (int i = 0; i < level; i++) {
      sb.append('\t');
    }
    if (this.isLeaf()) {
      // leaf node
      sb.append("Leaf[");
        sb.append(this.cluster.get(0).toShortString());

      sb.append("]");
    } else {
      sb.append("Branch[\n");
      level++;
      for (int i = 0; i < level; i++) {
        sb.append('\t');
      }
      sb.append('(').append(this.getLeftChild().toNString(level)).append(')')
          .append('(').append(this.getRightChild().toNString(level)).append(')');
      sb.append("]");
    }
    return sb.toString();
  }

  DendrogramNode<T> getMinNode() {
    return this.minNode;
  }

  Integer getMinDistance() {
    return this.minimumDistance;
  }

  public boolean isLeaf() {
    return this.isLeaf;
  }
}
