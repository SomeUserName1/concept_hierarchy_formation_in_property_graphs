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
    private List<DendrogramNode<T>> children = new ArrayList<>();
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
    private List<DendrogramNode<T>> minNode = new ArrayList<>();
    /**
     * flag to indicate wether the node is a leaf, thus has no children.
     */
    private boolean isLeaf;
    /**
     * storing the distance at which the two children were merged
     */
    private Float merged_distance;


    /**
     * @param children nodes that were merged in the previous step
     * @param distance distance between the nodes
     */
    DendrogramNode(List<DendrogramNode<T>> children, float distance) {
        this.isLeaf = false;

        this.children.addAll(children);

        for (DendrogramNode<T> child : this.children) {
            this.cluster.addAll(child.getCluster());
        }

        this.merged_distance = distance;
    }

    DendrogramNode(T b) {
        this.isLeaf = true;

        this.children = null;

        this.cluster.add(b);
    }

    /**
     * used for visualization (GUI).
     *
     * @return The left node that was merged to form this node
     */
    private List<DendrogramNode<T>> getChildren() {
        if (this.isLeaf) {
            throw new NullPointerException("A Leaf node has no children!");
        }
        return this.children;
    }

    /**
     * Used for debugging purposes
     *
     * @return the distances to all other dendrogram nodes in the current working set
     */
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
        checkMinNode(node, distance);

    }

    private void checkMinNode(DendrogramNode<T> node, float distance) {
        if (distance < this.minimumDistance) {
            this.minimumDistance = distance;
            this.minNode.clear();
            this.minNode.add(node);
        } else if (distance == minimumDistance) {
            this.minNode.add(node);
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
     * @param nodes to drop from the distance vector.
     */
    void dropDistances(List<DendrogramNode<T>> nodes) {
        for (DendrogramNode<T> node : nodes) {
            if (this.distances.get(node).equals(this.minimumDistance)) {
                this.minNode.remove(node);
            }
            this.distances.remove(node);
        }

        if (this.minNode.isEmpty()) {
            if (!this.distances.isEmpty()) {
                this.minimumDistance = Collections.min(this.distances.values());
                this.minNode.add(this.distances.entrySet().stream()
                        .filter(entry -> entry.getValue().equals(this.minimumDistance))
                        .map(Map.Entry::getKey).findFirst()
                        .orElseThrow(NullPointerException::new));
            } else {
                this.minimumDistance = Float.MAX_VALUE;
                this.minNode.clear();
            }
        }
    }

    private void setDistance(DendrogramNode<T> node, Float distance) {
        this.distances.put(node, distance);
        checkMinNode(node, distance);
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

            for (DendrogramNode<T> child : this.children) {
                child.print(level);
            }

        }
    }

    List<DendrogramNode<T>> getMinNode() {
        return this.minNode;
    }

    Float getMinDistance() {
        return this.minimumDistance;
    }

    public boolean isLeaf() {
        return this.isLeaf;
    }
}
