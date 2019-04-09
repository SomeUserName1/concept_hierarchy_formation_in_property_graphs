package preprocess.YelpBusiness;

import groovy.json.JsonSlurper;
import preprocess.DataLoader;

import java.io.*;
import java.util.*;

public class YelpBusinessDataLoader implements DataLoader {
    private ArrayList<YelpBusiness> data = new ArrayList<>();


    @Override
    public ArrayList<YelpBusiness> getData() {
        return data;
    }

    @Override
    public void read(String inputFilePath) {
        JsonSlurper parser = new JsonSlurper();
        try (BufferedReader br = new BufferedReader(new FileReader(inputFilePath))){
            String line = br.readLine();
            while ( line != null) {
                Map dataLine = (Map)parser.parseText(line);

                this.data.add(new YelpBusiness((String)dataLine.get("business_id"),
                        (String)dataLine.get("categories"),
                        (HashMap<String, String>)dataLine.get("attributes")));

                line = br.readLine();
            }
        }  catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void sample(int no_entries) {
        Random random = new Random();
        ArrayList<YelpBusiness> sample = new ArrayList<>();

        while(sample.size() < no_entries) {
            sample.add(this.data.get(random.nextInt()));
        }
        this.data = sample;
    }
}
