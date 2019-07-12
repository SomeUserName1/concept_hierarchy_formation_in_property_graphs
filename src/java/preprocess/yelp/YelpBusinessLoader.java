package preprocess.yelp;

import groovy.json.JsonSlurperClassic;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import preprocess.DataLoader;


/**
 * Loads data from the business.json file of the Yelp data set into a
 * YelpBusiness DataObject wrapper.
 * Provides functionality to sample from and filter the data
 */
public class YelpBusinessLoader extends DataLoader<YelpBusiness> {


  /**
   * reads data from the business.json file provided in the Yelp data set:
   * https://www.yelp.com/dataset
   * Stores the objects in a list as instance variable.
   * @param inputFilePath path to the business.json file
   */
  @SuppressWarnings("unchecked")
  public void read(String inputFilePath) {
    JsonSlurperClassic parser = new JsonSlurperClassic();
    try (BufferedReader br = new BufferedReader(new FileReader(inputFilePath))) {
      String line = br.readLine();
      YelpBusiness b;
      String categories;
      Map dataLine;
      while (line != null) {
        dataLine = (Map) parser.parseText(line);

        categories = (String) dataLine.get("categories");

        b =  categories != null ?
                new YelpBusiness((String) dataLine.get("business_id"), Arrays.asList(categories.split(",\\s+")),
                (HashMap<String, String>) dataLine.get("attributes"))
                : new YelpBusiness((String) dataLine.get("business_id"), null,
                (HashMap<String, String>) dataLine.get("attributes"));

        this.data.add(b);

        line = br.readLine();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    System.out.println("Finished reading data from Json");
  }

  /**
   * Shall filter the data set by a certain field and value (e.g. "category",
   * "restaurant"). Shall leave all entries having value in field.
   * @param field the field to consider. One of: business_id, categories,
   *              attributes
   * @param value the value to filter for. Must exist in at least one entry
   *              of the data set
   */
  public void filterBy(String field, String value) {
    System.out.println("Filtering by field: " + field + ", for value: " + value);
    List<YelpBusiness> filtered = new java.util.ArrayList<>();
    switch (field) {
      case "business_id":
        for (YelpBusiness b : this.data) {
          if (b.getBusinessId().equals(value)) {
            filtered.add(b);
          }
        }
        break;
      case "categories":
        for (YelpBusiness b : this.data) {
          if (b.getCategories().isEmpty()) {
            continue;
          }
          for (String category : b.getCategories()) {
            if (category.equals(value)) {
              filtered.add(b);
            }
          }
        }
        break;
      case "attributes":
        for (YelpBusiness b : this.data) {
          if (b.getAttributes().isEmpty()) {
            continue;
          }
          for (String attribute : b.getAttributes().keySet()) {
            if (attribute.equals(value)) {
              filtered.add(b);
            }
          }
        }
        break;
      default:
        throw new RuntimeException("Given field is not present in "
            + "YelpBusiness! Either extend the YelpBusiness class to include "
            + "it or specify a valid field");
    }
    this.data = filtered;
  }
}
