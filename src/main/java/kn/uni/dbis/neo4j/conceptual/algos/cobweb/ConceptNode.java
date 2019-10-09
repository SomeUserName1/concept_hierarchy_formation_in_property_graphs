package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

import org.apache.commons.lang.SerializationUtils;

import org.neo4j.graphdb.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import java.lang.Math;

public class ConceptNode implements Cloneable {
    private int count;
    private Map<String, Map<Value, Integer>> attributes;
    private ArrayList<ConceptNode> children;

    ConceptNode() {
        this.count = 0;
        this.attributes = new HashMap<>();
        this.children = new ArrayList<>();
    }

    void updateCounts(ConceptNode node) {
        this.count++;
        Map<Value, Integer> values;

        // loop over the properties of a N4J node
        for (Map.Entry<String, Map<Value, Integer>> fAttribs : node.getAttributes().entrySet()) {

            if (this.attributes.containsKey(fAttribs.getKey())) {
                // if the key was found in the concept description, look if we find matching values
                values = this.attributes.get(fAttribs.getKey());

                for (Map.Entry<Value, Integer> fVal : fAttribs.getValue().entrySet()) {
                    // Numeric attributes have only one NumericValue, i.e. the map is of size one

                    if (fVal.getKey() instanceof NumericValue) {
                        // Even though we have a NumericValue here, its not really mean and std but a single number with
                        // zero std
                        // TODO use welfords online algo.
                        Map.Entry<Value, Integer> num = values.entrySet().iterator().next();

                        NumericValue other = (NumericValue) fVal.getKey();
                        NumericValue numeric = (NumericValue) num.getKey();
                        int count = num.getValue();
                        double mean = numeric.getMean();
                        double newNr = other.getMean();
                        double variance = numeric.getStd() * numeric.getStd();

                        double newMean = mean + ((newNr - mean)/(count+1));
                        double newStd = Math.sqrt(variance + (((newNr - mean) * (newNr - newMean) - variance)/count+1));

                        numeric.setMean(newMean);
                        numeric.setStd(newStd);
                        num.setValue(count+1);
                    }

                    if (values.containsKey(fVal.getKey())) {
                        int vCount = values.get(fVal.getKey());
                        values.put(fVal.getKey(), vCount + 1);
                    } else {
                        values.put(fVal.getKey(), 1);
                    }
                }
            } else {
                HashMap<Value, Integer> map = new HashMap<>();
                map.put(Value.cast(fAttribs.getValue()), 1);
                attributes.put(fAttribs.getKey(), map);
            }
        }
    }

    void nodePropertiesToConcept(Node node) {
        // loop over the properties of a N4J node and cast them to a Value
        Object o;
        for (Map.Entry<String, Object> property : node.getAllProperties().entrySet()) {
            HashMap<Value, Integer> map = new HashMap<>();
            o = property.getValue();
            if (o.getClass().isArray()) {
                Object[] arr = (Object[])o;

                for(Object ob : arr) {
                    map.put(Value.cast(ob), 1);
                }
            } else {
                map.put(Value.cast(property.getValue()), 1);
            }
            attributes.put(property.getKey(), map);
        }
    }

    public int getCount() {
        return this.count;
    }

    public Map<String, Map<Value, Integer>> getAttributes() {
        return attributes;
    }

    public ArrayList<ConceptNode> getChildren() {
        return children;
    }

    public void addChild(ConceptNode node) {
        this.children.add(node);
    }
}

