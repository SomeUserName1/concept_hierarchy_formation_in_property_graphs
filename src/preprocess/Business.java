package preprocess;

import java.util.HashMap;
import java.util.Map;

public class Business {
    private String business_id;
    private String categories;
    private HashMap<String, String> attributes;

    public Business(String business_id, String categories, HashMap<String, String> attributes) {
        this.business_id = business_id;
        this.categories = categories;
        this.attributes = attributes;
    }

    public String getBusiness_id() {
        return business_id;
    }

    public String[] getCategories() {

        return categories.split(",");
    }

    public HashMap<String, String> getAttributes() {
        return attributes;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("Business ");
        str.append("id: ").append(this.business_id).append("\ncategories: ").append(this.categories)
                .append("\nattributes: \n");
        if (this.attributes == null)
            return str.append("None\n").toString();

        for (Map.Entry entry: this.attributes.entrySet())
            str.append("\t").append(entry);

        return str.append("\n").toString();
    }
}
