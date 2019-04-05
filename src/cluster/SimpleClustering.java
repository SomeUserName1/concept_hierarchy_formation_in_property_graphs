package cluster;

import preprocess.Business;

import java.util.*;

public class SimpleClustering {
    private int[][] distances;

    public SimpleClustering(ArrayList<Business> businesses) {
        this.distances = new int[businesses.size()][];
        calculateDistances(businesses);
        System.out.println(Arrays.deepToString(distances));
    }

    private void calculateDistances(ArrayList<Business> businesses) {
        for (int i = this.distances.length; i > 0; i--) {
            this.distances[i-1] = new int[i];
            System.out.println("outer index: " + i);
            for (int j = 0; j < i; j++) {
                this.distances[i-1][j] = compare(businesses.get(i-1), businesses.get(j));
            }
        }
    }

    private int compare(Business a, Business b) {
        if (a.getAttributes() == null && b.getAttributes() != null) {
            return b.getAttributes().keySet().size();
        } else if (a.getAttributes() != null && b.getAttributes() == null) {
            return a.getAttributes().keySet().size();
        } else if (a.getAttributes() == null && b.getAttributes() == null) {
            return 0;
        }

        Set<String> attribs_a = a.getAttributes().keySet();
        Set<String> attribs_b = b.getAttributes().keySet();

        // Union
        Set<String> symmetric_difference = new HashSet<>(attribs_a);
        symmetric_difference.addAll(attribs_b);

        Set<String> intersection = new HashSet<>(attribs_a);
        intersection.retainAll(attribs_b);

        symmetric_difference.removeAll(intersection);

        return symmetric_difference.size();
    }
}
