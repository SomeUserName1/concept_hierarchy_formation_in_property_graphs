package preprocess.synthetic;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class SyntheticNodeGenerator {
    private JSONArray nodeList;
    private int width;
    private int depth;

    @SuppressWarnings("unchecked")
    public SyntheticNodeGenerator(int width, int depth) {
        StringBuilder sb = new StringBuilder();
        this.width = width;
        this.depth = depth;
        int[] index = new int[depth];
        int amount = (int) Math.pow(width, depth);
        this.nodeList = new JSONArray();

        System.out.println(amount);
        for (int i = 0; i < amount; i++) {
            JSONObject syntheticNode = new JSONObject();
            syntheticNode.put("id", i);
            for (int j = 0; j < depth; j++) {
                sb.append("l");
                for (int k = 0; k < j+1; k++) {
                    sb.append(index[k]);
                }
                if (j < depth-1) sb.append(", ");
            }
            syntheticNode.put("labels", sb.toString());
            index = incrementIndex(index, depth-1);
            sb.setLength(0);
            this.nodeList.add(syntheticNode);
        }
    }

    @SuppressWarnings("unchecked")
    public void dehomogenize_levels() {
        Random rnd = new Random(42);
        int i = rnd.nextInt(this.depth);
        StringBuilder sb = new StringBuilder();

        for (Object node : this.nodeList) {

            List<String> labels = new ArrayList(Arrays.asList(
                ((String) ((JSONObject) node).get("labels")).split(",\\s+")));
            if (rnd.nextInt(3) % 2 == 0) labels.remove(i % this.depth);

            for (int j = 0; j < labels.size(); j++) {
                sb.append(labels.get(j));
                if (j < labels.size() - 1) sb.append(", ");
            }
            ((JSONObject) node).replace("labels", sb.toString());
            sb.setLength(0);
            i = rnd.nextInt(this.depth);
        }
    }

    @SuppressWarnings("unchecked")
    public void dehomogenize_names() {
        Random rnd = new Random(42);
        int i = rnd.nextInt(this.depth);
        StringBuilder sb = new StringBuilder();

        for (Object node : this.nodeList) {

            List<String> labels = new ArrayList(Arrays.asList(
                ((String) ((JSONObject) node).get("labels")).split(",\\s+")));

            for (int j = 0; j < labels.size(); j++) {
                sb.append(labels.get(j));
                // ASCII Digits start at 48 and reach till 57
                if (i == j) sb.append((char)(rnd.nextInt(10)+ 48));
                if (j < labels.size() - 1) sb.append(", ");
            }
            ((JSONObject) node).replace("labels", sb.toString());
            sb.setLength(0);
            i = rnd.nextInt(this.depth);
        }
    }

    @SuppressWarnings("unchecked")
    public void dehomogenize_cluster_size() {
        Random rnd = new Random(46);
        StringBuilder sb = new StringBuilder("l");
        int j = rnd.nextInt(this.depth);
        for (int i = 0; i < j; i++) {
            sb.append(rnd.nextInt(this.width));
        }
        String pruneBranch = sb.toString();
        System.out.println(pruneBranch);
        List<Object> toRemove = new ArrayList<>();

        for (Object node : this.nodeList) {

            List<String> labels = new ArrayList(Arrays.asList(
                ((String) ((JSONObject) node).get("labels")).split(",\\s+")));

            for (String elem : labels) {
                if (elem.equals(pruneBranch)) {
                    toRemove.add(node);
                }
            }
        }
        this.nodeList.removeAll(toRemove);
     }

    public void generate(String path) {
        //Write JSON file
        try (FileWriter file = new FileWriter(path + ".json")) {

            file.write(this.nodeList.toJSONString());
            file.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private int[] incrementIndex(int[] index, int level) {
        if (level < 0) return index;
        if (index[level] >= this.width-1) {
            index[level] = 0;
            return incrementIndex(index, level-1);
        } else {
            index[level] += 1;
            return index;
        }
    }

}
