package cluster.Dendrogram;

import cluster.Clustering;
import preprocess.YelpBusiness.YelpBusiness;

import java.util.*;

public class DendrogramClustering implements Clustering {
    private int[][] distances;
    private ArrayList<YelpBusiness> data;

    public DendrogramClustering(ArrayList<YelpBusiness> b) {
        this.data = b;
        calculateInitialDistances();
        cluster();
        System.out.println(Arrays.deepToString(distances));
    }



    public void calculateInitialDistances() {
        this.distances = new int[this.data.size()][];
        for (int i = this.distances.length; i > 0; i--) {
            this.distances[i-1] = new int[i];
            System.out.println("outer index: " + i);
            for (int j = 0; j < i; j++) {
                this.distances[i-1][j] = this.data.get(i-1).compare_attribs(this.data.get(j));
            }
        }
    }


    public void cluster() {
        merge();
        while
    }

    private int find_minimum_distance() {
        int min_idx_i = 0;
        int min_idx_j = 0;
        int min = Integer.MAX_VALUE;

        for (int i = 0; i < this.distances.length; i++) {
            for (int j = 0; j < this.distances[i].length; j++) {
                if (this.distances[i][j] < min) {
                    min_idx_i = i;
                    min_idx_j = j;
                }
            }
        }
        return
    }

    // Single Linkage Distance: D(C_i, C_j) = min_{p in C_i, q in C_j} (dist(p,q))

    // Complete Linkage Distance: D(C_i, C_j) = max_{p in C_i, q in C_j} (dist(p,q))
    

}
