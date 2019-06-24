import at.unisalzburg.dbresearch.apted.costmodel.CostModel;
import at.unisalzburg.dbresearch.apted.node.Node;
import cluster.Clustering;
import cluster.dendrogram.DendrogramClustering;
import preprocess.synthetic.SyntheticNode;
import preprocess.yelp.YelpBusiness;
import preprocess.yelp.YelpBusinessLoader;
import preprocess.synthetic.SyntheticNodeGenerator;
import preprocess.synthetic.SyntheticNodeLoader;
import at.unisalzburg.dbresearch.apted.costmodel.PerEditOperationStringNodeDataCostModel;
import at.unisalzburg.dbresearch.apted.distance.APTED;
import java.util.List;


public class Main {

  public static void main(String[] args) {

    //String path = "/home/someusername/snap/nextcloud-client/10/Nextcloud" +
      //  "/workspace/uni/8/bachelor/bachelor_project/data/synthetic";
    String path = "/home/fabian/Nextcloud/workspace/uni/8/bachelor" +
        "/bachelor_project/data/synthetic";
    SyntheticNodeGenerator sNG = new SyntheticNodeGenerator( 5, 4);
    sNG.generate(path);

    sNG.dehomogenize_cluster_size();
    sNG.generate(path + "_branch");

    sNG = new SyntheticNodeGenerator( 5, 4);
    sNG.dehomogenize_levels();
    sNG.generate(path + "_levels");

    sNG = new SyntheticNodeGenerator( 5, 4);
    sNG.dehomogenize_names();
    sNG.generate(path + "_names");
    sNG.dehomogenize_levels();
    sNG.dehomogenize_cluster_size();
    sNG.generate(path + "_all");

    CostModel costModel = new PerEditOperationStringNodeDataCostModel(1,1,0);
    APTED apted = new APTED<>(costModel);
    float cost = apted.computeEditDistance();
//    sNG.generate(path);
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


//    String fileName = path + "business.json";
//    YelpBusinessLoader yelpBusinessLoader = new YelpBusinessLoader();
//    yelpBusinessLoader.read(fileName);
//
//    yelpBusinessLoader.sample(100);
//
//    List<YelpBusiness> data = yelpBusinessLoader.getData();
//    System.out.println("Number of entries: " + data.size());
//
//    Clustering clustering = new DendrogramClustering<>(data, "single");
//    clustering.cluster();
//    ((DendrogramClustering) clustering).print();


  }
}
