package preprocess.synthetic;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class SyntheticNodeGenerator {

    @SuppressWarnings("unchecked")
    public void generate(String path, int width, int depth) {
        StringBuilder sb = new StringBuilder();
        int[] index = new int[depth];
        int amount = (int) Math.pow(width, depth);
        JSONArray nodeList = new JSONArray();

        System.out.println(amount);
        for (int i = 0; i < amount; i++) {
            JSONObject syntheticNode = new JSONObject();
            syntheticNode.put("id", i);
            for (int j = 0; j < depth; j++) {
                sb.append("l");
                for (int k = 0; k < j+1; k++) {
                    sb.append(index[k]);
                }
            }
            syntheticNode.put("labels", sb.toString());
            index = incrementIndex(index, width, depth-1);
            sb.setLength(0);
            nodeList.add(syntheticNode);
        }

        //Write JSON file
        try (FileWriter file = new FileWriter(path + "/synthetic.json")) {

            file.write(nodeList.toJSONString());
            file.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private int[] incrementIndex(int[] index, int width, int level) {
        if (level < 0) return index;
        if (index[level] >= width-1) {
            index[level] = 0;
            return incrementIndex(index, width, level-1);
        } else {
            index[level] += 1;
            return index;
        }
    }

}
