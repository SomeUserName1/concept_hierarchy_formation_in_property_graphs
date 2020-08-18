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
  private final JSONArray nodeList = new JSONArray();
  private final int width;
  private final int depth;
  private int idCount = 0;
  private final String path;

  @SuppressWarnings("unchecked")
  SyntheticNodeGenerator(String path, int width, int depth, int iter) {
    StringBuilder sb = new StringBuilder();
    this.width = width;
    this.depth = depth;
    int[] index = new int[depth];
    int amount = (int) Math.pow(width, depth);
    this.path = path;
    int iterLim = iter * amount;

    for (int step = 0; step < iterLim; step += amount) {
        for (int i = 0; i < amount; i++) {
          JSONObject syntheticNode = new JSONObject();
          syntheticNode.put("id", i + step);
          for (int j = 0; j < depth-1; j++) {
            sb.append("l");
            for (int k = 0; k < j+1; k++) {
              sb.append(index[k]);
            }
            if (j < depth-2) sb.append(", ");
          }
          syntheticNode.put("labels", sb.toString());
          index = incrementIndex(index, depth-1);
          sb.setLength(0);
          this.nodeList.add(syntheticNode);
        }
    }

    sb.setLength(0);
    sb.append("{Root");
    printBracketTree(sb, 0, 0);
    sb.append("}");

    try (FileWriter jsonFile = new FileWriter(path + "synthetic.json");
      FileWriter treeFile = new FileWriter(path + "synthetic.tree")) {

      //Write JSON file
      jsonFile.write(this.nodeList.toJSONString());
      jsonFile.flush();

      // Write Bracket notation style .tree file
      treeFile.write(sb.toString());
      treeFile.flush();

    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void printBracketTree(StringBuilder sb, int width, int depth) {
    if (depth == this.depth-1 || width == this.width) {
      return;
    }
    // print current node
    sb.append("{").append(this.idCount);
    this.idCount++;
    if (this.depth == depth + 1) {
      sb.append("}");
    }

    // print all children
    printBracketTree(sb, 0, depth + 1);
    // close the bracket and continue with next neighbour
    if (this.depth != depth + 1) {
      sb.append("}");
    }
    // print next element
    printBracketTree(sb, width + 1, depth);
  }

  // prob shall be in [0,1] and in steps of 1, 0.5, 0.33, 0.25, ...
  @SuppressWarnings("unchecked")
  void introduce_noise(boolean remLabels, boolean addLabels, boolean alterLabels, double prob) {
    StringBuilder sb = new StringBuilder();
    if (prob == 0) {
        try (FileWriter jsonFile = new FileWriter(this.path + "synthetic_noisy.json")) {
          //Write JSON file
          jsonFile.write(this.nodeList.toJSONString());
          jsonFile.flush();

        } catch (IOException e) {
          e.printStackTrace();
        }
        return;
    }
    int limit = Math.toIntExact(Math.round(1 / prob));
    Random rnd = new Random(limit + 42);

    for (Object node : this.nodeList) {
      int introduce = rnd.nextInt(limit);
      if (introduce != 0) {
        continue;
      }

      // Get the labels
      List<String> labels = new ArrayList<>(Arrays.asList(
          ((String) ((JSONObject) node).get("labels")).split(",\\s+")));
      if (remLabels) {
        labels.remove(rnd.nextInt(this.depth-1));
      }
      if (addLabels) {
        labels.add("l" + rnd.nextInt(1 << this.width + 1));
      }
      for (int j = 0; j < labels.size(); j++) {
        if (alterLabels && rnd.nextInt(this.depth-1) == j) {
          sb.append("l");
          int i = 0;
          do {
            sb.append((char) (rnd.nextInt(this.depth-1) + 48));
            ++i;
          } while (i < this.depth-1 && rnd.nextInt(2) == 0);
        } else {
          sb.append(labels.get(j));
        }
        if (j < labels.size() - 1) sb.append(", ");
      }

      ((JSONObject) node).replace("labels", sb.toString());
      sb.setLength(0);
    }

    try (FileWriter jsonFile = new FileWriter(this.path + "synthetic_noisy.json")) {
      //Write JSON file
      jsonFile.write(this.nodeList.toJSONString());
      jsonFile.flush();

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
