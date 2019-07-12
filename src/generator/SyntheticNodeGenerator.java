package generator;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

class SyntheticNodeGenerator {
    private JSONArray nodeList;
    private String bracketTree;
    private int width;
    private int depth;

    @SuppressWarnings("unchecked")
    SyntheticNodeGenerator(int width, int depth) {
        StringBuilder sb = new StringBuilder();
        this.width = width;
        this.depth = depth;
        int[] index = new int[depth];
        int amount = (int) Math.pow(width, depth);
        this.nodeList = new JSONArray();

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

        int k = 0;
        sb.setLength(0);
        sb.append("{Root");
        for (int j = 0; j < this.width; j++) {
            sb.append(k).append( "{");
            for (int i = 0; i < this.depth; i++) {

            }
            sb.append("}");
            k++;
        }
        sb.append()
    }


    // prob shall be in [0,1] and in steps of 1, 0.5, 0.33, 0.25, ...
    @SuppressWarnings("unchecked")
    void introduce_nose(boolean remLabels, boolean addLabels, boolean alterLabels, double prob) {
        StringBuilder sb = new StringBuilder();
        int limit = Math.toIntExact(Math.round(1 / prob));
        Random rnd = new Random(limit + 42);

        for (Object node : this.nodeList) {
            int introduce = rnd.nextInt(limit);
            if (introduce != 0) {
                continue;
            }

            // Get the labels
            List<String> labels = new ArrayList(Arrays.asList(
                ((String) ((JSONObject) node).get("labels")).split(",\\s+")));
            // why the 3? Randomly remove
            if (remLabels) {
                labels.remove(rnd.nextInt(this.depth));
            }
            if (addLabels) {
                labels.add("l" + rnd.nextInt(1 << this.width + 1));
            }
            for (int j = 0; j < labels.size(); j++) {
                if (alterLabels && rnd.nextInt(this.depth) == j) {
                    sb.append("l");
                    int i = 0;
                    do {
                        sb.append((char) (rnd.nextInt(this.depth) + 48));
                        ++i;
                    } while (i < this.depth && rnd.nextInt(2) == 0);
                } else {
                    sb.append(labels.get(j));
                }
                if (j < labels.size() - 1) sb.append(", ");
            }

            System.out.println("SB to replace the labels " + sb.toString());
            ((JSONObject) node).replace("labels", sb.toString());
            sb.setLength(0);
        }
    }


    void generate(String path) {

        try (FileWriter jsonFile = new FileWriter(path + ".json");
             FileWriter treeFile = new FileWriter(path + ".tree")) {
            //Write JSON file
            jsonFile.write(this.nodeList.toJSONString());
            jsonFile.flush();

            // Write Bracket notation style .tree file
            treeFile.write(this.bracketTree);
            treeFile.flush();

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
