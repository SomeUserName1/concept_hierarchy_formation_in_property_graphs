package cluster.dendrogram;

import cluster.DistanceFunction;

import java.util.*;

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
    private Map<DendrogramNode<T>, Float> distances = new HashMap<>();
    /**
     * current minimal distance.
     */
    private Float minimumDistance = Float.MAX_VALUE;
    /**
     * node having the current minimal distance.
     */
    private DendrogramNode<T> minNode = null;
    /**
     * flag to indicate wether the node is a leaf, thus has no children.
     */
    private boolean isLeaf;
    /**
     * storing the distance at which the two children were merged
     */
    private Float merged_distance;


    /**
     *
     * @param left
     * @param right
     * @param distance
     */
    DendrogramNode(DendrogramNode<T> left, DendrogramNode<T> right, float distance) {
        this.isLeaf = false;

        this.leftChild = left;
        this.rightChild = right;

        this.cluster.addAll(this.leftChild.getCluster());
        this.cluster.addAll(this.rightChild.getCluster());

        this.merged_distance = distance;
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
    private DendrogramNode<T> getLeftChild() {
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
    private DendrogramNode<T> getRightChild() {
        if (this.isLeaf) {
            throw new NullPointerException("A Leaf node has no children!");
        }
        return this.rightChild;
    }

    List<Float> getDistanceValues() {
        return new ArrayList<>(this.distances.values());
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
        if (node == this) {
            return;
        }
        float distance = distanceFunction.calculate(this, node);
        this.distances.put(node, distance);
        node.setDistance(this, distance);
        if (distance < this.minimumDistance) {
            this.minimumDistance = distance;
            this.minNode = node;
        }
    }


    Float getDistance(DendrogramNode node) {
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
                                .filter(entry -> entry.getValue().equals(this.minimumDistance))
                                .map(Map.Entry::getKey).findFirst()
                                .orElseThrow(NullPointerException::new);
            } else {
                this.minimumDistance = Float.MAX_VALUE;
                this.minNode = null;
            }
        }
    }

    private void setDistance(DendrogramNode<T> node, Float distance) {
        this.distances.put(node, distance);
        if (distance < this.minimumDistance) {
            this.minimumDistance = distance;
            this.minNode = node;
        }
    }

    void print(int level) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < level; i++) {
            sb.append('\t');
        }
        if (this.isLeaf) {
            sb.append("Leaf(").append(this.cluster.get(0).toString())
                    .append(")");
            System.out.println(sb);
        } else {
            level++;
            sb.append(this.merged_distance).append(" Node");
            System.out.println(sb);
            this.getLeftChild().print(level);
            this.getRightChild().print(level);
        }
    }

    DendrogramNode<T> getMinNode() {
        return this.minNode;
    }

    Float getMinDistance() {
        return this.minimumDistance;
    }

    public boolean isLeaf() {
        return this.isLeaf;
    }
}