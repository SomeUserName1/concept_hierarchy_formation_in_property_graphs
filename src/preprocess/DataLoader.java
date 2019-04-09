package preprocess;

import preprocess.YelpBusiness.YelpBusiness;

import java.util.ArrayList;

public interface DataLoader {
    ArrayList<YelpBusiness> getData();

    void read(String inputFilePath);

    void sample(int no_entries);
}
