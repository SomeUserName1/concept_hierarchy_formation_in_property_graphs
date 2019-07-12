package generator;

import java.util.*;

public class SyntheticNode {
    private int id;
    private List<String> labels;

    public SyntheticNode(int id, List<String> labels) {
        this.id = id;
        this.labels = labels;
    }

    public List<String> getLabels() {
        return this.labels;
    }

    private float jaccardDistance(SyntheticNode b) {
        if (this.labels.isEmpty() || b.getLabels().isEmpty()) {
            return 0;
        }

        List<String> union = new ArrayList<>(this.labels);

        Set<String> intersection = new HashSet<>(union);
        intersection.retainAll(b.getLabels());

        union.addAll(b.getLabels());


        return 1.0f - ((float)intersection.size())/union.size() + 0.0001f; // (min + 0.00001f );
    }

    /**
     * Compares the attributes of two YelpBusinesses by keys.
     * @param d A DataObject; Must be a yelp or throws an exception
     * @return The number of distinct attributes by key (only) aka the
     *        symmetric difference between the key sets
     */
    public float compare(SyntheticNode d) {
        return jaccardDistance(d);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("SyntheticNode");
        str.append("id: ").append(this.id).append(", labels: ");
        if (this.labels == null) {
            return str.append("None").toString();
        }

        for (String entry : this.labels) {
            str.append(entry).append(", ");
        }

        return str.toString();
    }

    public String toShortString() {
        return String.valueOf(this.id);
    }
}
