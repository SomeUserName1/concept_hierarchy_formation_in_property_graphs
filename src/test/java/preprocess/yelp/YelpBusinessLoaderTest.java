package preprocess.yelp;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class YelpBusinessLoaderTest {
  @Test
  public void testRead() {
    YelpBusinessLoader loader = new YelpBusinessLoader();
    loader.read("/home/fabian/Nextcloud/workspace/uni/8" +
        "/bachelor_project/data/business0.json");
    List<YelpBusiness> data = loader.getData();
    assertEquals(data.size(), 10);
    YelpBusiness b = data.get(0);
    assertEquals(b.getBusinessId(), "ABjONdA5Fw8XBOM65tmW4w");
    assertEquals(b.getCategories(),
        Arrays.asList("Plumbing", " Water Heater Installation/Repair",
        " Professional Services", " Home Services", " Contractors"));
    assertEquals(b.getAttributes(),
        new HashMap<String, String>() {{
            put("BusinessAcceptsCreditCards", "True");
              put("ByAppointmentOnly", "True");
          }});
  }

  @Test
  public void testNullCategories() {
    // todo
    assertEquals(0, 1);
  }

  @Test
  public void testNullAttributes() {
    // todo
    assertEquals(0,1);
  }

  @Test
  public void testSample() {
    YelpBusinessLoader loader = new YelpBusinessLoader();
    loader.read("/home/fabian/Nextcloud/workspace/uni/8" +
        "/bachelor_project/data/business0.json");
    loader.sample(3);
    List<YelpBusiness> data = loader.getData();
    assertEquals(data.size(), 3);
  }

  @Test
  public void testFilterBusinessId() {
    YelpBusinessLoader loader = new YelpBusinessLoader();
    loader.read("/home/fabian/Nextcloud/workspace/uni/8" +
        "/bachelor_project/data/business0.json");
    loader.filterBy("business_id", "c1f_VAX1KIK8-JoVhjbYOw");
    List<YelpBusiness> data = loader.getData();
    assertEquals(data.size(), 1);
  }

  @Test
  public void testFilterCategories() {
    YelpBusinessLoader loader = new YelpBusinessLoader();
    loader.read("/home/fabian/Nextcloud/workspace/uni/8" +
        "/bachelor_project/data/business0.json");
    loader.filterBy("categories", "Books");
    List<YelpBusiness> data = loader.getData();
    assertEquals(data.size(), 1);
  }

  @Test
  public void testFilterAttributes() {
    YelpBusinessLoader loader = new YelpBusinessLoader();
    loader.read("/home/fabian/Nextcloud/workspace/uni/8" +
        "/bachelor_project/data/business0.json");
    loader.filterBy("attributes", "RestaurantsPriceRange2");
    List<YelpBusiness> data = loader.getData();
    assertEquals(data.size(), 6);
  }
}
