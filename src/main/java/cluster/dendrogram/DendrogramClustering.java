package cluster.dendrogram;

import cluster.Clustering;
import cluster.DistanceFunction;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import preprocess.DataObject;
import visualize.DendrogramJPanel;


// TODO: Specification, Tests, rework access, evaluate

/**
 * Implements hierarchical linkage-based clustering.
 * Pseudo code:
 * 1. Initial clusters consisting of individual nodes, calculate distances
 * for all pairs of clusters.
 * 2. Merge closest clusters
 * 3. Update distance of cluster to all others
 * 4. If there is only one cluster left, terminate, else goto 2. 
 * Leaf Nodes: contain a single Object of the data type -> Initial loss matrix
 * Inner Nodes: contain union of all children -> recalculate loss vector based
 * on linkage convention
 */
public class DendrogramClustering<T extends DataObject> implements Clustering {
  private final List<DendrogramNode<T>> workingSet =
      new ArrayList<>();
  private DistanceFunction<T> distanceFunction;

  /**
   * Initializes the clustering algorithm by wrapping all DataObjects into
   * Dendrogram nodes to form the initial clusters, thus adding them to the
   * current working set.
   * @param bs List of DataObjects to cluster
   */
  public DendrogramClustering(List<T> bs) {
    for (T b : bs) {
      DendrogramNode<T> leaf = new DendrogramNode<>(b);
      workingSet.add(leaf);
    }
    this.distanceFunction = new LinkageDistance<>("single");
  }

  /**
   * compute the distances between the initial clusters.
   */
  private void initializeDistances() {
    for (DendrogramNode<T> n1 : this.workingSet) {
      for (DendrogramNode<T> n2 : this.workingSet) {
        if (n1 != n2 && n1.getDistance(n2) == null) {
          n1.calculateDistance(n2, this.distanceFunction);
        }
      }
    }
  }

  /**
   * Cluster the DataObjects by:
   *  1. find the minimal distance between two clusters
   *  2. merge those clusters to a new node, add the new node and remove the
   *     merged nodes from the working set. Also drop the entries from each
   *     nodes distance vector
   *  3. compute the distances from the new node to all others in the working
   *      set
   */
  public void cluster() {
    System.out.println("Starting to cluster");

    initializeDistances();

    int i = 0;
    boolean removeMerged0;
    boolean removeMerged1;

    while (this.workingSet.size() > 1) {
      //find_minimum_distance
      DendrogramNode<T> containingNode = null;
      int minimumDistance = Integer.MAX_VALUE;
      for (DendrogramNode<T> node : this.workingSet) {
        if (node.getMinDistance() < minimumDistance) {
          containingNode = node;
          minimumDistance = node.getMinDistance();
        }
      }

      //merge
      removeMerged0 = this.workingSet.remove(containingNode);
      removeMerged1 =
          this.workingSet.remove(containingNode.getMinNode());

      assert (removeMerged0 && removeMerged1);

      DendrogramNode<T> newNode = new DendrogramNode<>(containingNode,
          containingNode.getMinNode());

      for (DendrogramNode<T> node : this.workingSet) {
        node.dropDistance(containingNode);
        node.dropDistance(containingNode.getMinNode());
      }

      this.workingSet.add(newNode);

      // update distances
      for (DendrogramNode<T> node : this.workingSet) {
        if (node == newNode) {
          continue;
        }
        newNode.calculateDistance(node, this.distanceFunction);
      }

      System.out.println("Working Set size: " + this.workingSet.size());
      System.out.println("Finished iteration " + i++);
    }
    System.out.println("Finished clustering");
  }

  @Override
  public void evaluate() {
  }

  /**
   * Assumes clustering has finished
   */
  public void print() {
    System.out.println(this.workingSet.get(0).toNString(0));
  }

  @Override
  public void visualize() {
    JFrame jframe = new JFrame();
    jframe.add(new DendrogramJPanel<>(this.workingSet));
    jframe.setSize(500, 500);
    jframe.setVisible(true);
  }
}
