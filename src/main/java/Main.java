import cluster.Clustering;
import cluster.dendrogram.DendrogramClustering;
import preprocess.synthetic.SyntheticNode;
import preprocess.yelp.YelpBusiness;
import preprocess.yelp.YelpBusinessLoader;
import preprocess.synthetic.SyntheticNodeGenerator;
import preprocess.synthetic.SyntheticNodeLoader;


import java.util.List;


public class Main {

  public static void main(String[] args) {

    String path = "/home/someusername/snap/nextcloud-client/10/Nextcloud/workspace/uni/8/bachelor/bachelor_project/data/";
//    SyntheticNodeGenerator sNG = new SyntheticNodeGenerator();
//    sNG.generate(path, 5, 4);
//    String fileName = path + "synthetic.json";
//    SyntheticNodeLoader syntheticNodeLoader = new SyntheticNodeLoader();
//    syntheticNodeLoader.read(fileName);
//    List<SyntheticNode> data = syntheticNodeLoader.getData();
//
//    System.out.println("Number of entries: " + data.size());
//
//    Clustering clustering = new DendrogramClustering<>(data, "complete");
//    clustering.cluster();
//    ((DendrogramClustering) clustering).print();


    String fileName = path + "business.json";
    YelpBusinessLoader yelpBusinessLoader = new YelpBusinessLoader();
    yelpBusinessLoader.read(fileName);

    yelpBusinessLoader.sample(100);

    List<YelpBusiness> data = yelpBusinessLoader.getData();
    System.out.println("Number of entries: " + data.size());

    Clustering clustering = new DendrogramClustering<>(data, "single");
    clustering.cluster();
    ((DendrogramClustering) clustering).print();


  }
}
