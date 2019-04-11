package cluster;

import cluster.dendrogram.DendrogramNode;
import preprocess.DataObject;

@FunctionalInterface
public interface DistanceFunction<T extends DataObject> {
  Integer calculate(DendrogramNode<T> node1, DendrogramNode<T> node2);
}
