package cluster.Dendrogram;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @param <T>
 */
public interface DendrogramNode<T> {

    T getLeftChild();

    T getRightChild();

    ArrayList getCluster();

    // Single Linkage Distance: D(C_i, C_j) = min_{p in C_i, q in C_j} (dist(p,q))
    void calculateSLDistance(T node);

    // Complete Linkage Distance: D(C_i, C_j) = max_{p in C_i, q in C_j} (dist(p,q))
    void calculateCLDistance(T node);

    List<Integer> getDistance();
}
