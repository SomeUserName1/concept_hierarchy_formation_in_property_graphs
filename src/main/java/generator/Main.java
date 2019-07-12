package generator;


public class Main {

  public static void main(String[] args) {
    String path = null;
    int depth = 0;
    int width = 0;
    boolean remLabels = false;
    boolean addLabels = false;
    boolean alterLabels = false;
    double probability = 0;
    boolean[] valid = new boolean[7];


    String helpMessage = "\n" +
        "Generate synthetic data with hierarchical label structure and versions with additional noise.\n" +
        "\n" +
        "SYNTAX\n" +
        "\n" +
        "    java -jar generateSyntheticData.jar -p  -d  -w  -n -pr \n" +
        "\n" +
        "    java -jar generateSyntheticData.jar -h\n" +
        "\n" +
        "OPTIONS\n" +
        "\n" +
        "    -h, --help \n" +
        "        print this help message.\n" +
        "\n" +
        "    -p path,\n" +
        "        specifies the output directory.\n" +
        "\n" +
        "    -d height \n" +
        "        The height of the hierarcy.\n" +
        "\n" +
        "    -w width,\n" +
        "        the depth of the hierarcy.\n" +
        "\n" +
        "    -n remLabels addLabels alterLabels\n" +
        "        specifies the output directory.\n" +
        "\n" +
        "    -pr probability \n" +
        "        the probability of noise being introduced.\n" +
        "\n";


    try {
      for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
          case "-p":
          case "--path":
            path = args[i + 1];
            i = i + 1;
            valid[0] = true;
            break;
          case "-d":
          case "--depth":
            depth =  Integer.valueOf(args[i + 1]);
            i = i + 1;
            valid[1] = true;
            break;
          case "-w":
          case "--width":
            width =  Integer.valueOf(args[i + 1]);
            i = i + 1;
            valid[2] = true;
            break;
          case "-n":
          case "--noise":
            remLabels = Integer.valueOf(args[i+1]) == 1;
            addLabels = Integer.valueOf(args[i+2]) == 1;
            alterLabels = Integer.valueOf(args[i+3]) == 1;
            i = i + 3;
            valid[3] = true;
            valid[4] = true;
            valid[5] = true;
            break;
          case "-pr":
          case "--probability":
            probability =  Double.valueOf(args[i + 1]);
            i = i + 1;
            valid[6] = true;
            break;
          default:
            System.out.println(helpMessage);
            System.exit(0);
        }
      }
    } catch (ArrayIndexOutOfBoundsException e) {
      System.out.println("Too few arguments.");
      System.exit(0);
    }

    for (boolean entry : valid) {
      if (!entry) {
        System.out.println("Didn't specify a necessary parameter");
        System.out.println(helpMessage);
        System.exit(0);
      }
    }
    System.out.println("Executing synthetic data generation\n"
    + "Noise parameters:" + remLabels + " " + addLabels + " " + alterLabels + " prob "
        + probability);
    SyntheticNodeGenerator sNG = new SyntheticNodeGenerator(path, width, depth);
    sNG.introduce_noise(remLabels, addLabels, alterLabels, probability);
  }
}
