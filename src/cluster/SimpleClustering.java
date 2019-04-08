package cluster;

import preprocess.Business;

import java.util.*;

public class SimpleClustering {
    private int[][] distances;
    private ArrayList<Business> sample;

    public SimpleClustering(ArrayList<Business> businesses) {
        sampling(businesses);
        calculateDistances();
        cluster();
        System.out.println(Arrays.deepToString(distances));
    }

    private void sampling(ArrayList<Business> b) {
        
    }

    private void calculateDistances() {
        this.distances = new int[this.sample.size()][];
        for (int i = this.distances.length; i > 0; i--) {
            this.distances[i-1] = new int[i];
            System.out.println("outer index: " + i);
            for (int j = 0; j < i; j++) {
                this.distances[i-1][j] = this.sample.get(i-1).compare_attribs(this.sample.get(j));
            }
        }
    }


    private void cluster() {

    }


}
