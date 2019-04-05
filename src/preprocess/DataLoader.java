package preprocess;

import groovy.json.JsonSlurper;
import java.io.*;
import java.util.*;

public class DataLoader {
    public ArrayList<Business> getData() {
        return data;
    }

    private ArrayList<Business> data = new ArrayList<>();


    public void readBusinesses(String inputFilePath) {
        JsonSlurper parser = new JsonSlurper();
        try (BufferedReader br = new BufferedReader(new FileReader(inputFilePath))){
            String line = br.readLine();
            while ( line != null) {
                extract((Map)parser.parseText(line));
                line = br.readLine();
            }
        }  catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void extract(Map data) {
        this.data.add(new Business((String)data.get("business_id"),
                (String)data.get("categories"),
                (HashMap<String, String>)data.get("attributes")));
    }
}
