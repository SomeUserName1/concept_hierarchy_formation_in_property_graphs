package preprocess.YelpBusiness;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class YelpBusiness {
    private String business_id;
    private String categories;
    private HashMap<String, String> attributes;

    public YelpBusiness(String business_id, String categories, HashMap<String, String> attributes) {
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

    public int compare_attribs(YelpBusiness b) {
        if (this.getAttributes() == null && b.getAttributes() != null) {
            return b.getAttributes().keySet().size();
        } else if (this.getAttributes() != null && b.getAttributes() == null) {
            return this.getAttributes().keySet().size();
        } else if (this.getAttributes() == null && b.getAttributes() == null) {
            return 0;
        }

        Set<String> attribs_a = this.getAttributes().keySet();
        Set<String> attribs_b = b.getAttributes().keySet();

        // Union
        Set<String> symmetric_difference = new HashSet<>(attribs_a);
        symmetric_difference.addAll(attribs_b);

        Set<String> intersection = new HashSet<>(attribs_a);
        intersection.retainAll(attribs_b);

        symmetric_difference.removeAll(intersection);

        return symmetric_difference.size();
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("YelpBusiness ");
        str.append("id: ").append(this.business_id).append("\ncategories: ").append(this.categories)
                .append("\nattributes: \n");
        if (this.attributes == null)
            return str.append("None\n").toString();

        for (Map.Entry entry: this.attributes.entrySet())
            str.append("\t").append(entry);

        return str.append("\n").toString();
    }
}
