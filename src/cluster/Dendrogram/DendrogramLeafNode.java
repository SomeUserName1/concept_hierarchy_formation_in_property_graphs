package cluster.Dendrogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DendrogramLeafNode<T> implements DendrogramNode<T> {
    private ArrayList<T> cluster = new ArrayList<>();
    private Map<DendrogramNode, Integer> distances = new HashMap<>();

    public DendrogramLeafNode(T element) {
        this.cluster.add(element);
    }

    @Override
    public T getLeftChild() {
        throw new NullPointerException("A Leaf node has no children!");
    }

    @Override
    public T getRightChild() {
        throw new NullPointerException("A Leaf node has no children!");
    }

    @Override
    public ArrayList<T> getCluster() {
        return this.cluster;
    }

    @Override
    public void calculateSLDistance(T node) {
        // TODO here: Calculate distance to the node.
        // FIXME: Store distances per node? makes sense as it belongs to the cluster
        //
    }

    @Override
    public void calculateCLDistance(T node) {

    }

    @Override
    public List<Integer> getDistance() {
        return null;
    }

}
