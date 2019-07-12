package preprocess;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// TODO: Specification
public abstract class DataLoader<T extends DataObject> {
  protected List<T> data = new ArrayList<>();

  public List<T> getData() {
    return this.data;
  }

  public abstract void read(String inputFilePath);

  /**
   * Samples a certain amount of entries from the data set.
   * @param noEntries the number of samples that shall be left after sampling
   */
  public void sample(int noEntries) {
    if (noEntries > this.data.size()) {
      return;
    }
    Random random = new Random();
    List<T> sample = new ArrayList<>();
    int s = this.data.size();
    while (sample.size() < noEntries) {
      sample.add(this.data.get(random.nextInt(s)));
    }
    this.data = sample;
    System.out.println("Finished sampling");
  }

  public abstract void filterBy(String field, String value);
}
