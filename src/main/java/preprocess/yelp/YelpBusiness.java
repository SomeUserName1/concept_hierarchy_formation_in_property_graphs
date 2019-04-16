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

  private float relativeSymmetricDifference(YelpBusiness b) {
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
    Set<String> union = new HashSet<>(attributesA);
    union.addAll(attributesB);

    Set<String> intersection = new HashSet<>(attributesA);
    intersection.retainAll(attributesB);

    union.removeAll(intersection);

    return ((float)union.size())/(attributesA.size() + attributesB.size() + 0.0001f);
  }

  private float relativeIntersection(YelpBusiness b) {
      if (this.getAttributes() == null || b.getAttributes() == null) {
        return 0;
      }

      Set<String> attributesA = this.getAttributes().keySet();
      Set<String> attributesB = b.getAttributes().keySet();

      Set<String> intersection = new HashSet<>(attributesA);
      intersection.retainAll(attributesB);

      float min = attributesA.size() < attributesB.size() ? attributesA.size() : attributesB.size();

    return 1.0f - ((float)intersection.size())/(min + 0.00001f );
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

    float k = 0.5f;
    //return symmetricDifference(b);
    return k * relativeSymmetricDifference(b) + (1-k) * relativeIntersection(b);
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
