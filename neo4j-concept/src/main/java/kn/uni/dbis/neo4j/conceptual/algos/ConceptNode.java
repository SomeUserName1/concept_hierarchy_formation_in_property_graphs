package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

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
public class ConceptNode {
  /**
   * Attribute aggregation over all (sub-)instances of this Concept.
   */
  private final Map<String, List<Value>> attributes;

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
  public ConceptNode(String name) {
    this.count = 0;
    this.label = name;
    this.attributes = new HashMap<>();
    this.children = new ArrayList<>();
    this.parent = null;
  }

  /**
   * Constructor.
   */
  public ConceptNode() {
    this.count = 1;
    this.label = null;
    this.attributes = new HashMap<>();
    this.children = new ArrayList<>();
    this.parent = null;
  }

  /**
   * Copy constructor.
   *
   * @param node ConceptNode to copy.
   */
  public ConceptNode(final ConceptNode node) {
    this.count = node.count;
    this.label = node.label;
    this.attributes = new HashMap<>();

    List<Value> values;
    String attributeName;
    for (Map.Entry<String, List<Value>> attribute : node.attributes.entrySet()) {
      attributeName = attribute.getKey();
      values = new ArrayList<>();
      this.attributes.put(attributeName, values);
      for (Value value : attribute.getValue()) {
        values.add(value.copy());
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

    List<Value> values = new ArrayList<>();

    if (propertyContainer instanceof Relationship) {
      final Relationship rel = (Relationship) propertyContainer;
      values.add(new NominalValue(rel.getType().name()));
      this.attributes.put("RelType", values);
    } else if (propertyContainer instanceof Node) {
      final Node mNode = (Node) propertyContainer;
      for (Label label : mNode.getLabels()) {
        values.add(new NominalValue(label.name()));
        this.attributes.put("Label", values);
      }
    }

    // loop over the properties of a N4J node and cast them to a Value
    Object o;
    Object[] arr;
    for (Map.Entry<String, Object> property : propertyContainer.getAllProperties().entrySet()) {
      values = new ArrayList<>();
      o = property.getValue();
      if (o.getClass().isArray()) {
        arr = (Object[]) o;

        for (Object ob : arr) {
          values.add(Value.cast(ob));
        }
      } else {
        values.add(Value.cast(property.getValue()));
      }
      this.attributes.put(property.getKey(), values);
    }
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof ConceptNode) {
      final ConceptNode node = (ConceptNode) o;
      return node.count == this.count && node.attributes.equals(this.attributes);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.count, this.attributes);
  }

  /**
   * Aggregates the count and attributes of the nodes, hosted by this concept.
   *
   * @param node  the node to incorporate into the concept.
   */
  public void updateCounts(final ConceptNode node) {
    List<Value> thisValues;
    NumericValue thisNumeric;
    boolean matched;

    this.count++;
    // loop over the attributes of the node to incorporate
    for (Map.Entry<String, List<Value>> otherAttributes : node.getAttributes().entrySet()) {
      thisValues = this.attributes.get(otherAttributes.getKey());
      // If the attribute is present in this node, check against the present values
      if (thisValues != null) {
        for (Value otherValue : otherAttributes.getValue()) {
          // iterate over values
          if (otherValue instanceof NumericValue) {
            // When encountering a NumericValue, it must be matched with the one present in this node
            matched = false;
            for (Value thisVal : thisValues) {
              // per attribute there is only one NumericValue that accumulates all numeric values.
              if (thisVal instanceof NumericValue) {
                thisNumeric = (NumericValue) thisVal;
                thisNumeric.update(otherValue);
                matched = true;
                break;
              }
            }
            // When there is no NumericValue in this node for the attribute, add the one of the other node
            if (!matched) {
              thisValues.add(otherValue.copy());
            }
          } else {
            final int idx = thisValues.indexOf(otherValue);
            if (idx == -1) {
              thisValues.add(otherValue.copy());
            } else {
              thisValues.get(idx).update(otherValue);
            }
          }
        }
      } else {
        // Else add the attribute and it's properties of the new node as they are
        final List<Value> copies = new ArrayList<>();
        for (Value value : otherAttributes.getValue()) {
          copies.add(value.copy());
        }
        this.attributes.put(otherAttributes.getKey(), copies);
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
    } else if (this.parent.equals(c)) {
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
  public Map<String, List<Value>> getAttributes() {
    return this.attributes;
  }

  /**
   * Getter for the sub-concepts of this node.
   *
   * @return the sub-concepts of this node
   */
  public List<ConceptNode> getChildren() {
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
    sb.append("-|_");
    for (int i = 0; i < depth; i++) {
      sb.append("¸__");
    }
    sb.append(this.toString()).append("\n");

    for (ConceptNode child : this.children) {
      child.printRec(sb, depth + 1);
    }

    return sb.toString();
  }

  @Override
  public String toString() {
    return "ConceptNode " + System.identityHashCode(this) + "Count: " + this.count + " Attributes: "
        + this.attributes.toString();
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
