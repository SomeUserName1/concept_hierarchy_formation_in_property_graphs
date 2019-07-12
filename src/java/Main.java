import preprocess.synthetic.SyntheticNode;
import preprocess.yelp.YelpBusiness;
import preprocess.yelp.YelpBusinessLoader;
import preprocess.synthetic.SyntheticNodeGenerator;
import preprocess.synthetic.SyntheticNodeLoader;
import java.util.List;


public class Main {

  public static void main(String[] args) {
    String path = "/home/fabian/Nextcloud/workspace/uni/8/bachelor" +
        "/bachelor_project/data/synthetic";
    SyntheticNodeGenerator sNG = new SyntheticNodeGenerator( 5, 4);
    sNG.generate(path + ".json");

    sNG.dehomogenize_cluster_size();
    sNG.generate(path + "_branch.json");

    sNG = new SyntheticNodeGenerator( 5, 4);
    sNG.dehomogenize_levels();
    sNG.generate(path + "_levels.json");

    sNG = new SyntheticNodeGenerator( 5, 4);
    sNG.dehomogenize_names();
    sNG.generate(path + "_names.json");
    sNG.dehomogenize_levels();
    sNG.dehomogenize_cluster_size();
    sNG.generate(path + "_all.json");

    CostModel costModel = new PerEditOperationStringNodeDataCostModel(1,1,0);
    APTED apted = new APTED<>(costModel);
    float cost = apted.computeEditDistance();

    String fileName = path + "business.json";
    YelpBusinessLoader yelpBusinessLoader = new YelpBusinessLoader();
    yelpBusinessLoader.read(fileName);

    yelpBusinessLoader.sample(1000);



  }
}
