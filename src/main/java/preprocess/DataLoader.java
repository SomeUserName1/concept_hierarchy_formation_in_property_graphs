package preprocess;

import java.util.List;

// TODO: Specification
public interface DataLoader<T extends DataObject> {
  List<T> getData();

  void read(String inputFilePath);

  void sample(int no_entries);

  // FIXME use sth else than strings;
  void filterBy(String field, String value);
}
