package preprocess.yelp;

import org.junit.Test;
import preprocess.DataObject;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class YelpBusinessTest {
  final private static String exampleBid1 = "123";
  final private static List<String> exampleCategories1 = Arrays.asList("Tex",
      "Mex");
  final private static HashMap<String, String> exampleAttributes1 =
      new HashMap<String, String>() {{
        put("RestaurantsPriceRange", "2");
        put("BikeParking", "True");
        put("BusinessParking", "{'garage': False, 'street': False, " +
            "'validated': False, 'lot': False, 'valet': False}");
      }};

  final private static String exampleBid2 = "1234";
  final private static List<String> exampleCategories2 = Arrays.asList(
      "Burgers", "Bar");
  final private static HashMap<String, String> exampleAttributes2 =
      new HashMap<String, String>() {{
        put("RestaurantsPriceRange", "2");
        put("GoodForChildren", "True");
        put("BusinessParking", "{'garage': False, 'street': False, " +
            "'validated': False, 'lot': False, 'valet': False}");
      }};

  final private static String exampleBid3 = "12";
  final private static List<String> exampleCategories3 = Arrays.asList(
      "CarWash", "Pit");
  final private static HashMap<String, String> exampleAttributes3 =
      new HashMap<String, String>() {{
        put("FuelPrice", "2");
        put("CarWashWithWax", "True");
        put("SthStupid", "{'garage': False, 'street': False, " +
            "'validated': False, 'lot': False, 'valet': False}");
      }};

  @Test
  public void compareToSelf() {
    YelpBusiness b = new YelpBusiness(exampleBid1, exampleCategories1,
        exampleAttributes1);
    assertEquals(b.compare(b), 0);
  }

  @Test
  public void compareDistanceYelpBusiness() {
    YelpBusiness b1 = new YelpBusiness(exampleBid1, exampleCategories1,
        exampleAttributes1);
    YelpBusiness b2 = new YelpBusiness(exampleBid2, exampleCategories2,
        exampleAttributes2);
    YelpBusiness b3 = new YelpBusiness(exampleBid3, exampleCategories3,
        exampleAttributes3);
    // b1 and b2 overlap in 2 of 3 attributes, b1 and b3 are distinct
    assertTrue(b1.compare(b2) < b1.compare(b3));
  }

  @Test
  public void compareFullyDistinctAttributes() {
    YelpBusiness b1 = new YelpBusiness(exampleBid1, exampleCategories1,
        exampleAttributes1);
    YelpBusiness b3 = new YelpBusiness(exampleBid3, exampleCategories3,
        exampleAttributes3);

    assertEquals(b1.compare(b3), 6);
  }

  @Test
  public void compareOverlappingYelpBusiness() {
    YelpBusiness b1 = new YelpBusiness(exampleBid1, exampleCategories1,
        exampleAttributes1);
    YelpBusiness b2 = new YelpBusiness(exampleBid2, exampleCategories2,
        exampleAttributes2);

    assertEquals(b1.compare(b2), 2);
  }

  @Test(expected = RuntimeException.class)
  public void compareAnotherDataObject() {
    YelpBusiness b2 = new YelpBusiness(exampleBid2, exampleCategories2,
        exampleAttributes2);
    StubDataObject d = new StubDataObject();

    b2.compare(d);
  }

  @Test
  public void compareNullAttributesOne() {
    YelpBusiness b1 = new YelpBusiness(exampleBid1, exampleCategories1,
        exampleAttributes1);
    YelpBusiness b2 = new YelpBusiness(exampleBid2, exampleCategories2,
        null);

    assertEquals(b1.compare(b2), b1.getAttributes().keySet().size());
    assertEquals(b2.compare(b1), b1.getAttributes().keySet().size());
  }

  @Test
  public void compareNullAttributesBoth() {
    YelpBusiness b1 = new YelpBusiness(exampleBid1, exampleCategories1,
        null);
    YelpBusiness b2 = new YelpBusiness(exampleBid2, exampleCategories2,
        null);
    assertEquals(b1.compare(b2), 0);
  }

  class StubDataObject implements DataObject {

    @Override
    public int compare(DataObject d) {
      return 0;
    }
  }
}
