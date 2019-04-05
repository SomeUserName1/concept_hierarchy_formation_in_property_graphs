import cluster.SimpleClustering;
import preprocess.DataLoader;


public class Main {

    public static void main(String[] args) {

        // TODO set up args & interfaces

        String fileName = "/home/fabian/Nextcloud/workspace/uni/8/bachelor_project/data/business0.json";
        DataLoader dataLoader = new DataLoader();
        dataLoader.readBusinesses(fileName);

        SimpleClustering clustering = new SimpleClustering(dataLoader.getData());

        /*List<Point> points = null;
        try {
            points = dataLoader.readPoints("dataset/" + fileName);
        } catch (IOException e) {
            System.out.println("Could not read input file: " + fileName);
            System.exit(1);
        }

        System.out.println("Number of points: " + points.size());

        // Run Chameleon algorithm
        Chameleon chameleon = new Chameleon(k, initNrOfClusters, resultNrOfClusters, points);
        List<Cluster> clusters = chameleon.run();

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
