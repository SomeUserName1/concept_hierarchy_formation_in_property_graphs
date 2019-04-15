package cluster.dendrogram;

import cluster.DistanceFunction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import preprocess.DataObject;

/**
 * Distance functions for linkage-based hierarchical clustering.
 * Currently implements single, average and complete linkage
 * @param <T> DataObject to use the compare function of.
 */
class LinkageDistance<T extends DataObject> implements DistanceFunction<T> {
  /** One of single, average or compete. */
  private String type;

  LinkageDistance(String which) {
    this.type = which;
  }

  /**
   * Calculates the distance, dependent on the type that was passed to the
   * constructor.
   * @param node1 first cluster/node to compare with
   * @param node2 second cluster/node
   * @return the distance between the nodes/clusters
   */
  @Override
  public Float calculate(DendrogramNode<T> node1, DendrogramNode<T> node2) {
    List<Float> distances = new ArrayList<>();
    for (T element1 : node1.getCluster()) {
      for (T element2 : node2.getCluster()) {
        distances.add(element1.compare(element2));
      }
    }
    switch (this.type) {
      case "single":
        return Collections.min(distances);
      case "complete":
        return Collections.max(distances);
      case "average":
        return (float) distances.stream().mapToDouble(a -> a).average()
            .orElse(0);
      default:
        throw new NoSuchElementException("only single, complete and average "
            + "are currently available");
    }
  }
}
