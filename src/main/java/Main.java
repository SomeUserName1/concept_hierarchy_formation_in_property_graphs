import cluster.Clustering;
import cluster.dendrogram.DendrogramClustering;
import preprocess.yelp.YelpBusiness;
import preprocess.yelp.YelpBusinessDataLoader;

import java.util.ArrayList;
import java.util.List;


public class Main {

  public static void main(String[] args) {

    // TODO set up args & interfaces

    String fileName = "/home/someusername/snap/nextcloud-client/10/Nextcloud/workspace/uni/8/bachelor_project/data/" +
            "business.json";
    YelpBusinessDataLoader yelpBusinessDataLoader = new YelpBusinessDataLoader();
    yelpBusinessDataLoader.read(fileName);

    yelpBusinessDataLoader.filterBy("categories", "Food");
    yelpBusinessDataLoader.sample(4000);

    List<YelpBusiness> data = yelpBusinessDataLoader.getData();
    System.out.println("Number of entries: " + data.size());

    Clustering clustering = new DendrogramClustering<>(data, "average");
    clustering.cluster();
    ((DendrogramClustering) clustering).print();

        /*List<Point> points = null;
        try {
            points = yelpBusinessDataLoader.readPoints("dataset/" + fileName);
        } catch (IOException e) {
            System.out.println("Could not read input file: " + fileName);
            System.exit(1);
        }

        System.out.println("Number of points: " + points.size());

        // Run chameleon algorithm
        chameleon chameleon = new chameleon(k, initNrOfClusters, resultNrOfClusters, points);
        List<Clustering> clusters = chameleon.run();

        // Compute metrics
        MetricsCalculator metricsCalculator = new MetricsCalculator();
        Metrics metrics = metricsCalculator.calculate(clusters);

        // Visualize results
        ClusteringVisualization visualization = new ClusteringVisualization(clusters);
        visualization.drawImage(fileName.replace(".csv", ".png"));

        // Print metrics
        System.out.println("Metrics: ");
        System.out.println(metrics);
        clusters.forEach(x -> System.out.println(x.printMetrics()));*/
  }
}
