package preprocess.synthetic;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import preprocess.DataLoader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SyntheticNodeLoader extends DataLoader<SyntheticNode> {
    private List<SyntheticNode> data = new ArrayList<>();

    public List<SyntheticNode> getData() {
        return this.data;
    }

    @Override
    public void read(String inputFilePath) {
//JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();

        try (FileReader reader = new FileReader("employees.json"))
        {
            //Read JSON file
            Object obj = jsonParser.parse(reader);

            JSONArray syntheticNodeList = (JSONArray) obj;
            System.out.println(syntheticNodeList);

            //Iterate over employee array
            for (Object object : syntheticNodeList) {
                JSONObject node = (JSONObject) object;
                List<String> labels = Arrays.asList(((String) node.get("labels")).split(","));
                data.add(new SyntheticNode((int)node.get("id"), labels));
            }

        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void sample(int no_entries) {

    }

    @Override
    public void filterBy(String field, String value) {

    }
}
