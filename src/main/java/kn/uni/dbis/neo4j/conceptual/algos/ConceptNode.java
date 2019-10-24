package kn.uni.dbis.neo4j.conceptual.algos;

import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.PropertyContainer;
import org.neo4j.graphdb.Relationship;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import java.lang.Math;

public class ConceptNode implements Cloneable {
    private int count;
    private Map<String, Map<Value, Integer>> attributes;
    private ArrayList<ConceptNode> children;
    private ConceptNode parent;

    public ConceptNode() {
        this.count = 0;
        this.attributes = new HashMap<>();
        this.children = new ArrayList<>();
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof ConceptNode) {
            ConceptNode node = (ConceptNode)o;

            return node.count == this.count && node.attributes.equals(this.attributes)
                    && (new HashSet<>(node.children).equals(new HashSet<>(this.children)));
        } else {
            return false;
        }
    }

    public ConceptNode clone() {
        try {
            super.clone();
        } catch (final CloneNotSupportedException e) {
            e.printStackTrace();
        }
        ConceptNode clone = new ConceptNode();
        clone.count = this.count;
        for (Map.Entry<String, Map<Value, Integer>> attribute : this.attributes.entrySet()) {
            String attributeName = attribute.getKey();
            Map<Value, Integer> values = new HashMap<>();
            this.attributes.put(attributeName, values);
            for (Map.Entry<Value, Integer> value : attribute.getValue().entrySet()) {
                values.put(value.getKey().clone(), value.getValue());
            }
        }
        clone.children = new ArrayList<>(this.children);

        return clone;
    }


    void updateCounts(ConceptNode node, boolean merge) {
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

                        Map.Entry<Value, Integer> num = values.entrySet().iterator().next();

                        NumericValue other = (NumericValue) fVal.getKey();
                        NumericValue numeric = (NumericValue) num.getKey();
                        int count = num.getValue();
                        double mean = numeric.getMean();
                        double otherMean = other.getMean();


                        if (merge) {
                            int otherCount = fVal.getValue();
                            int totalCount = (count + otherCount);
                            numeric.setMean((count * mean + otherCount * otherMean) / totalCount);
                            numeric.setStd((count * numeric.getStd() + otherCount * other.getStd()) / totalCount);

                        } else {
                            // Even though we have a NumericValue here, its not really mean and std but a single number with
                            // zero std for the non-merge case
                            // TODO chens online algo
                            double variance = numeric.getStd() * numeric.getStd();
                            double newMean = mean + ((otherMean - mean) / (count + 1));
                            double newStd = Math.sqrt(variance + (((otherMean - mean) * (otherMean - newMean) - variance) / count + 1));

                            numeric.setMean(newMean);
                            numeric.setStd(newStd);
                            num.setValue(count + 1);
                        }
                    } else {
                        if (values.containsKey(fVal.getKey())) {
                            int vCount = values.get(fVal.getKey());
                            values.put(fVal.getKey(), vCount + fVal.getValue());
                        } else {
                            values.put(fVal.getKey(), fVal.getValue());
                        }
                    }
                }
            } else {
                values = new HashMap<>();
                this.attributes.put(fAttribs.getKey(), values);
                for (Map.Entry<Value, Integer> value : fAttribs.getValue().entrySet()) {
                    values.put(value.getKey().clone(), value.getValue());
                }
            }
        }
    }


    // Orders are ignored, so all lists are converted to sets
    public void propertyContainerToConceptNode(PropertyContainer propertyContainer) {

        HashMap<Value, Integer> map = new HashMap<>();

        if (propertyContainer instanceof Relationship) {
            Relationship rel = (Relationship)propertyContainer;
            map.put(new NominalValue(rel.getType().name()), 1);
            attributes.put("Type", map);
            map = new HashMap<>();
            map.put(new NominalValue(rel.getId()),1);
            attributes.put("RelationshipID", map);
        } else if (propertyContainer instanceof Node) {
            Node mNode = (Node)propertyContainer;
            for (Label label :mNode.getLabels()) {
                map.put(new NominalValue(label.name()), 1);
                attributes.put("Label", map);
                map = new HashMap<>();
                map.put(new NominalValue(mNode.getId()),1);
                attributes.put("RelationshipID", map);
            }
        }

        // loop over the properties of a N4J node and cast them to a Value
        Object o;
        for (Map.Entry<String, Object> property : propertyContainer.getAllProperties().entrySet()) {
            map = new HashMap<>();
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

    boolean isSuperConcept(ConceptNode c) {
        if (this.parent == null) {
            return false;
        }
        if (this.parent.equals(c)) {
            return true;
        } else {
            return parent.isSuperConcept(c);
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

    void addChild(ConceptNode node) {
        this.children.add(node);
    }

    public ConceptNode getParent() {
        return parent;
    }

    public void setParent(ConceptNode parent) {
        this.parent = parent;
    }
}

