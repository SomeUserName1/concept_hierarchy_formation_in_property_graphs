package cluster.Dendrogram;

import java.util.ArrayList;
import java.util.List;

public class DendrogramInnerNode<T> implements DendrogramNode<T> {

    @Override
    public T getLeftChild() {
        return null;
    }

    @Override
    public T getRightChild() {
        return null;
    }

    @Override
    public ArrayList getCluster() {
        return null;
    }

    @Override
    public void calculateSLDistance() {

    }

    @Override
    public void calculateCLDistance() {

    }

    @Override
    public List<Integer> getDistance() {
        return null;
    }
}
