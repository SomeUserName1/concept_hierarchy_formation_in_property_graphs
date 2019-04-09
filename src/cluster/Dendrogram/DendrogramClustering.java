package cluster.Dendrogram;

import cluster.Clustering;
import preprocess.YelpBusiness.YelpBusiness;

import java.util.*;

/**
 * Implements hierarchical linkage-based clustering.
 *
 * Pseudo code:
 *  1. Initial clusters consisting of individual nodes, calculate distances
 *      for all pairs of clusters.
 *  2. Merge closest clusters
 *  3. Update distance of cluster to all others
 *  4. If there is only one cluster left, terminate, else goto 2.
 *
 *  Leaf Nodes: contain a single Object of the data type -> Initial loss matrix
 *  Inner Nodes: contain union of all children -> recalculate loss vector based
 *      on linkage convention
 */
public class DendrogramClustering implements Clustering {
    private ArrayList<DendrogramNode> working_set = new ArrayList<>();

    public DendrogramClustering(ArrayList<YelpBusiness> b) {
        for (YelpBusiness yb : b) {
            DendrogramNode<YelpBusiness> leaf = new DendrogramLeafNode<>(yb);
            working_set.add(leaf);
        }
        cluster();
    }



    public void calculateInitialDistances() {
        // TODO do this per nodes and in the nodes;
/*        this.distances = new int[this.data.size()][];
        for (int i = this.distances.length; i > 0; i--) {
            this.distances[i-1] = new int[i];
            for (int j = 0; j < i; j++) {
                this.distances[i-1][j] = this.data.get(i-1).compare_attribs(this.data.get(j));
            }
        }*/
    }


    public void cluster() {
        while (this.working_set.size() > 1) {
            //merge
            //update_distances
        }
    }

    private void merge() {
        // TODO
    }

    private void update_distance() {
        // TODO
    }

    private List<Integer> find_minimum_distance() {
        // TODO
        return null;
    }


}
