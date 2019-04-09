import cluster.Clustering;
import cluster.Dendrogram.DendrogramClustering;
import preprocess.YelpBusiness.YelpBusinessDataLoader;

public class Main {

    public static void main(String[] args) {

        // TODO set up args & interfaces

        String fileName = "/home/fabian/Nextcloud/workspace/uni/8/bachelor_project/data/business.json";
        YelpBusinessDataLoader yelpBusinessDataLoader = new YelpBusinessDataLoader();
        yelpBusinessDataLoader.read(fileName);
        yelpBusinessDataLoader.sample(10);

        Clustering clustering = new DendrogramClustering(yelpBusinessDataLoader.getData());

        /*List<Point> points = null;
        try {
            points = yelpBusinessDataLoader.readPoints("dataset/" + fileName);
        } catch (IOException e) {
            System.out.println("Could not read input file: " + fileName);
            System.exit(1);
        }

        System.out.println("Number of points: " + points.size());

        // Run Chameleon algorithm
        Chameleon chameleon = new Chameleon(k, initNrOfClusters, resultNrOfClusters, points);
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
