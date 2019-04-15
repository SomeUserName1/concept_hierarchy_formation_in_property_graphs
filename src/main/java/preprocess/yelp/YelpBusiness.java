package preprocess.yelp;

import java.util.*;

import preprocess.DataObject;

/**
 * Data Object wrapper for the Yelp businesses data set entries.
 * Assumptions:
 * - Only the businessId, categories, attributes fields are
 * considered as a first proof of concept.
 * - compare implements a comparison over the attributes field, as it's
 * semi-structured so that it can be used as proof of concept.
 */
public class YelpBusiness implements DataObject {
  private String businessId;
  private List<String> categories;
  private Map<String, String> attributes;

  YelpBusiness(String businessId, List<String> categories, HashMap<String,
      String> attributes) {
    this.businessId = businessId;
    if (categories != null) {
      this.categories = categories;
    } else {
      this.categories = new ArrayList<>();
    }
    if (attributes != null) {
      this.attributes = attributes;
    } else {
      this.attributes = new HashMap<>();
    }
  }

  String getBusinessId() {
    return businessId;
  }

  List<String> getCategories() {
    return categories;
  }

  Map<String, String> getAttributes() {
    return attributes;
  }

  public int symmetricDifference(YelpBusiness b) {
    if (this.getAttributes() == null && b.getAttributes() != null) {
      return b.getAttributes().keySet().size();
    } else if (this.getAttributes() != null && b.getAttributes() == null) {
      return this.getAttributes().keySet().size();
    } else if (this.getAttributes() == null) {
      return 0;
    }

    Set<String> attributesA = this.getAttributes().keySet();
    Set<String> attributesB = b.getAttributes().keySet();

    // Union
    Set<String> symmetricDifference = new HashSet<>(attributesA);
    symmetricDifference.addAll(attributesB);

    Set<String> intersection = new HashSet<>(attributesA);
    intersection.retainAll(attributesB);

    symmetricDifference.removeAll(intersection);

    return symmetricDifference.size();
  }

  private float intersection(YelpBusiness b) {
      if (this.getAttributes() == null || b.getAttributes() == null) {
        return 0;
      }

      Set<String> attributesA = this.getAttributes().keySet();
      Set<String> attributesB = b.getAttributes().keySet();

      Set<String> intersection = new HashSet<>(attributesA);
      intersection.retainAll(attributesB);

      return intersection.size() > 0 ? 1.0f/intersection.size() : 2;
  }

  /**
   * Compares the attributes of two YelpBusinesses by keys.
   * @param d A DataObject; Must be a yelp or throws an exception
   * @return The number of distinct attributes by key (only) aka the
   *        symmetric difference between the key sets
   */
  public float compare(DataObject d) throws RuntimeException {
    YelpBusiness b;
    if ((d.toString().contains("YelpBusiness"))) {
      b = (YelpBusiness) d;
    } else {
      throw new RuntimeException("You can't compare two different data "
          + "objects. If you want to, create a common Wrapper Object that"
          + " implements compare appropriately for both DataObjects.");
    }
    //return symmetricDifference(b);
    return intersection(b);
  }

  @Override
  public String toString() {
    StringBuilder str = new StringBuilder("YelpBusiness ");
    str.append("id: ").append(this.businessId, 1, 5).append(", attributes: ");
    if (this.attributes == null) {
      return str.append("None").toString();
    }

    List<String> attributeKeys = new ArrayList<>(this.attributes.keySet());
    Collections.sort(attributeKeys);
    for (String entry : attributeKeys) {
      str.append(entry).append(", ");
    }

    return str.toString();
  }

  public String toShortString() {
    return this.getBusinessId().substring(1, 5);
  }

}
