package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.PropertyContainer;
import org.neo4j.graphdb.Relationship;

// TODO names/labels

/**
 * The basic data type for the conceptual hierarchy (tree) constructed and used by cobweb.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class ConceptNode implements Cloneable {
  /**
   * Attribute aggregation over all (sub-)instances of this Concept.
   */
  private final Map<String, Map<Value, AtomicInteger>> attributes;

  /**
   * Number of Nodes under this ConceptNode.
   */
  private int count;
  /**
   * Assigned label of the Concept.
   */
  private String label;
  /**
   * The sub-concepts of this Concept.
   */
  private ArrayList<ConceptNode> children;
  /**
   * The super-concept of this concept.
   */
  private ConceptNode parent;

  /**
   * Named constructor, also initializing it's name.
   */
  public ConceptNode() {
    this.count = 1;
    this.label = null;
    this.attributes = new HashMap<>();
    this.children = new ArrayList<>();
    this.parent = null;
  }

  /**
   * .computeIfAbsent(key, k -> new AtomicInteger()).incrementAndGet();
   * Copy constructor.
   *
   * @param node ConceptNode to copy.
   */
  ConceptNode(final ConceptNode node) {
    this.count = node.count;
    this.label = node.label;
    this.attributes = new HashMap<>();

    Map<Value, AtomicInteger> values;
    String attributeName;
    for (Map.Entry<String, Map<Value, AtomicInteger>> attribute : node.attributes.entrySet()) {
      attributeName = attribute.getKey();
      values = new HashMap<>();
      this.attributes.put(attributeName, values);
      for (Map.Entry<Value, AtomicInteger> value : attribute.getValue().entrySet()) {
        values.put(value.getKey().clone(), value.getValue());
      }
    }

    this.children = new ArrayList<>();
    this.children = new ArrayList<>(node.children);
    this.parent = node.parent;
  }

  /**
   * Converts a property container into a Singleton ConceptNode.
   * Orders are ignored, so all lists are converted to sets.
   *
   * @param propertyContainer The property container to parse.
   */
  ConceptNode(final PropertyContainer propertyContainer) {
    this.count = 1;
    this.attributes = new HashMap<>();
    this.children = new ArrayList<>();
    this.parent = null;

    HashMap<Value, AtomicInteger> map = new HashMap<>();

    if (propertyContainer instanceof Relationship) {
      final Relationship rel = (Relationship) propertyContainer;
      map.put(new NominalValue(rel.getType().name()), new AtomicInteger(1));
      this.attributes.put("Type", map);
    } else if (propertyContainer instanceof Node) {
      final Node mNode = (Node) propertyContainer;
      for (Label label : mNode.getLabels()) {
        map.put(new NominalValue(label.name()), new AtomicInteger(1));
        this.attributes.put("Label", map);
      }
    }

    // loop over the properties of a N4J node and cast them to a Value
    Object o;
    Object[] arr;
    for (Map.Entry<String, Object> property : propertyContainer.getAllProperties().entrySet()) {
      map = new HashMap<>();
      o = property.getValue();
      if (o.getClass().isArray()) {
        arr = (Object[]) o;

        for (Object ob : arr) {
          map.put(Value.cast(ob), new AtomicInteger(1));
        }
      } else {
        map.put(Value.cast(property.getValue()), new AtomicInteger(1));
      }
      this.attributes.put(property.getKey(), map);
    }
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof ConceptNode) {
      final ConceptNode node = (ConceptNode) o;

      return node.count == this.count && node.attributes.equals(this.attributes)
          && new HashSet<>(node.children).equals(new HashSet<>(this.children));
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.count, this.attributes, this.label);
  }

  /**
   * Aggregates the count and attributes of the nodes, hosted by this concept.
   *
   * @param node  the node to incorporate into the concept.
   * @param merge if the node is a singleton concept or needs to be propperly merged.
   */
  public void updateCounts(final ConceptNode node, final boolean merge) {
    Map<Value, AtomicInteger> values;
    NumericValue other;
    NumericValue numeric;
    Map.Entry<Value, AtomicInteger> num;
    AtomicInteger count;
    double mean;
    double otherMean;
    AtomicInteger otherCount;
    AtomicInteger totalCount;
    double variance;
    double newMean;
    double newStd;

    this.count++;
    // loop over the properties of a N4J node
    for (Map.Entry<String, Map<Value, AtomicInteger>> fAttribs : node.getAttributes().entrySet()) {

      if (this.attributes.containsKey(fAttribs.getKey())) {
        // if the key was found in the concept description, look if we find matching values
        values = this.attributes.get(fAttribs.getKey());

        for (Map.Entry<Value, AtomicInteger> fVal : fAttribs.getValue().entrySet()) {
          // Numeric attributes have only one NumericValue, i.e. the map is of size one

          if (fVal.getKey() instanceof NumericValue) {
            num = values.entrySet().iterator().next();
            other = (NumericValue) fVal.getKey();
            numeric = (NumericValue) num.getKey();
            count = num.getValue();
            mean = numeric.getMean();
            otherMean = other.getMean();


            if (merge) {
              // TODO double check
              otherCount = fVal.getValue();
              totalCount = count + otherCount;
              numeric.setMean(count * mean + otherCount * otherMean / totalCount);
              numeric.setStd((count * numeric.getStd() + otherCount * other.getStd()) / totalCount);
              count.addAndGet(otherCount.get());
            } else {
              // Even though we have a NumericValue here, its not really mean and std but a single
              // number with zero std for the non-merge case
              // TODO chans online algo
              variance = numeric.getStd() * numeric.getStd();
              newMean = mean + ((otherMean - mean) / (count + 1));
              newStd = Math.sqrt(variance + (((otherMean - mean) * (otherMean - newMean) - variance)
                  / count + 1));

              numeric.setMean(newMean);
              numeric.setStd(newStd);
              num.setValue(count + 1);
            }
          } else {
            values.computeIfAbsent(fVal.getKey(), k -> new AtomicInteger(1));
          }
        }
      } else {
        values = new HashMap<>();
        this.attributes.put(fAttribs.getKey(), values);
        for (Map.Entry<Value, AtomicInteger> value : fAttribs.getValue().entrySet()) {
          values.put(value.getKey().clone(), value.getValue());
        }
      }
    }
  }

  /**
   * Check if the given concept is a superconcept of the current.
   *
   * @param c ConceptNode the candidate concept.
   * @return true if c is the parent of the node or one of it's parent.
   */
  boolean isSuperConcept(final ConceptNode c) {
    if (this.parent == null) {
      return false;
    }
    if (this.parent.equals(c)) {
      return true;
    } else {
      return this.parent.isSuperConcept(c);
    }
  }

  /**
   * Getter for the count.
   *
   * @return number of instances and sub-concepts hosted by this concept
   */
  int getCount() {
    return this.count;
  }

  /**
   * Getter for the Attribute Value aggregation map.
   *
   * @return the map that stores the attributes as strings and possible values with counts as map
   */
  public Map<String, Map<Value, AtomicInteger>> getAttributes() {
    return this.attributes;
  }

  /**
   * Getter for the sub-concepts of this node.
   *
   * @return the sub-concepts of this node
   */
  ArrayList<ConceptNode> getChildren() {
    return this.children;
  }

  /**
   * Add the given ConceptNode as child.
   *
   * @param node the node to add to the children.
   */
  void addChild(final ConceptNode node) {
    this.children.add(node);
  }

  /**
   * Getter for the super-concept.
   *
   * @return the super concept of this node
   */
  public ConceptNode getParent() {
    return this.parent;
  }

  /**
   * Setter for the super-concept.
   *
   * @param parent the super-concept to be set
   */
  void setParent(final ConceptNode parent) {
    this.parent = parent;
  }

  /**
   * Prints nodes recursively from this node downwards the tree.
   *
   * @param sb    StringBuilder to use.
   * @param depth the depth when called in order to arrange the output appropriately
   * @return a String holding the representation of the tree
   */
  String printRec(final StringBuilder sb, final int depth) {
    for (int i = 0; i < depth; i++) {
      sb.append("|__");
    }
    sb.append(this.toString());

    for (ConceptNode child : this.children) {
      child.printRec(sb, depth + 1);
    }

    return sb.toString();
  }

  @Override
  public String toString() {
    return "Count: " + this.count + " Attributes: " + this.attributes.toString();
  }

  /**
   * Getter fot the label.
   *
   * @return the label
   */
  public String getLabel() {
    return this.label;
  }

  /**
   * Setter fot the label.
   *
   * @param label the label to be set
   */
  void setLabel(final String label) {
    this.label = label;
  }
}
