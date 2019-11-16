package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import apoc.atomic.Atomic;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.PropertyContainer;
import org.neo4j.graphdb.Relationship;
import scala.xml.Atom;

/**
 * The basic data type for the conceptual hierarchy (tree) constructed and used by cobweb.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class ConceptNode {
  /**
   * Attribute aggregation over all (sub-)instances of this Concept.
   */
  private final ConcurrentMap<String, List<Value>> attributes;

  /**
   * Number of Nodes under this ConceptNode.
   */
  private AtomicInteger count;
  /**
   * Neo4j ID of the Incorporated node..
   */
  private AtomicReference<String> id;

  /**
   * Label of the node.
   */
  private AtomicReference<String> label;

  /**
   * The sub-concepts of this Concept.
   */
  private List<ConceptNode> children;
  /**
   * The super-concept of this concept.
   */
  private AtomicReference<ConceptNode> parent;

  /**
   * Constructor.
   */
  public ConceptNode() {
    this.count = new AtomicInteger(1);
    this.id = new AtomicReference<>(null);
    this.attributes = new ConcurrentHashMap<>();
    this.children = Collections.synchronizedList(new ArrayList<>());
    this.parent = new AtomicReference<>(null);
    this.label = new AtomicReference<>(null);
  }

  /**
   * Copy constructor.
   *
   * @param node ConceptNode to copy.
   */
  public ConceptNode(final ConceptNode node) {
    this.count = new AtomicInteger(node.count.get());
    this.id = new AtomicReference<>(node.id.get());
    this.attributes = new ConcurrentHashMap<>();
    this.label = new AtomicReference<>(null);

    List<Value> values;
    String attributeName;
    for (Map.Entry<String, List<Value>> attribute : node.attributes.entrySet()) {
      attributeName = attribute.getKey();
      values = Collections.synchronizedList(new ArrayList<>());
      this.attributes.put(attributeName, values);
      for (Value value : attribute.getValue()) {
        values.add(value.copy());
      }
    }

    this.children = Collections.synchronizedList(new ArrayList<>(node.children));
    this.parent = new AtomicReference<>(node.parent.get());
  }

  /**
   * Converts a property container into a Singleton ConceptNode.
   * Orders are ignored, so all lists are converted to sets.
   *
   * @param propertyContainer The property container to parse.
   */
  ConceptNode(final PropertyContainer propertyContainer) {
    this.count = new AtomicInteger(1);
    this.attributes = new ConcurrentHashMap<>();
    this.children = Collections.synchronizedList(new ArrayList<>());
    this.parent = new AtomicReference<>(null);
    this.label = new AtomicReference<>(null);

    List<Value> values = Collections.synchronizedList(new ArrayList<>());

    if (propertyContainer instanceof Relationship) {
      final Relationship rel = (Relationship) propertyContainer;
      this.id = new AtomicReference<>(Long.toString(rel.getId()));
      values.add(new NominalValue(rel.getType().name()));
      this.attributes.put("RelType", values);
    } else if (propertyContainer instanceof Node) {
      final Node mNode = (Node) propertyContainer;
      this.id = new AtomicReference<>(Long.toString(mNode.getId()));
      for (Label label : mNode.getLabels()) {
        values.add(new NominalValue(label.name()));
        this.attributes.put("Label", values);
      }
    }

    // loop over the properties of a N4J node and cast them to a Value
    Object o;
    Object[] arr;
    for (Map.Entry<String, Object> property : propertyContainer.getAllProperties().entrySet()) {
      values = Collections.synchronizedList(new ArrayList<>());
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

  public synchronized ConceptNode root() {
    this.count.set(0);
    this.parent.set(this);

    return this;
  }

  /**
   * Aggregates the count and attributes of the nodes, hosted by this concept.
   *
   * @param node  the node to incorporate into the concept.
   */
  public synchronized void updateCounts(final ConceptNode node) {
    List<Value> thisValues;
    NumericValue thisNumeric;
    boolean matched;

    this.count.set(this.count.get() + node.count.get());
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
        final List<Value> copies = Collections.synchronizedList(new ArrayList<>());
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
    if (this.parent.get() == this) {
      return false;
    } else if (this.parent.get().equals(c)) {
      return true;
    } else {
      return this.parent.get().isSuperConcept(c);
    }
  }

  /**
   * Getter for the count.
   *
   * @return number of instances and sub-concepts hosted by this concept
   */
  public int getCount() {
    return this.count.get();
  }

  /**
   * Setter for the count.
   *
   * @param count number of instances and sub-concepts hosted by this concept
   */
  void setCount(final int count) {
    this.count.set(count);
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
    return this.parent.get();
  }

  /**
   * Setter for the super-concept.
   *
   * @param parent the super-concept to be set
   */
  void setParent(final ConceptNode parent) {
    this.parent.set(parent);
  }

  /**
   * Prints nodes recursively from this node downwards the tree.
   *
   * @param sb    StringBuilder to use.
   * @param depth the depth when called in order to arrange the output appropriately
   * @return a String holding the representation of the tree
   */
  String printRec(final StringBuilder sb, final int depth, int max_depth) {
    if (depth == 0) {
      sb.append("|__");
    } else {
      for (int i = 0; i < depth; i++) {
        sb.append("\t");
      }
      sb.append("|____");
    }

    sb.append(this.toString()).append("\n");
    if (depth <= max_depth) {
      for (ConceptNode child : this.children) {
        child.printRec(sb, depth + 1, max_depth);
      }
    }

    return sb.toString();
  }


  @Override
  public boolean equals(final Object o) {
    if (o instanceof ConceptNode) {
      final ConceptNode node = (ConceptNode) o;
      return node.count.get() == this.count.get() && node.attributes.equals(this.attributes);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.count.get(), this.attributes);
  }

  @Override
  public String toString() {
    String id = this.id.get() != null ? "ID: " + this.id.get() : "";
    return "ConceptNode " + id + " Count: " + this.count.get() + " Attributes: "
        + this.attributes.toString();
  }


  public String toTexTable() {
    final StringBuilder sb = new StringBuilder();
    sb.append("ConceptNode \\hspace{1cm} P(this) = ")
        .append(Double.toString((double) this.count.get() / (double) parent.get().count.get()), 0, 5)
        .append("\\\\")
        .append("Attributes: \\\\")
        .append("\\begin{tabular}{|c|c|c|} \\hline");
    for (Map.Entry<String, List<Value>> attribute : this.attributes.entrySet()) {
      sb.append("\\multirow{4}{*}{").append(attribute.getKey()).append("} ");
      for (Value value : attribute.getValue()) {
        sb.append(value.toTexString()).append((double)value.getCount()/(double)this.count.get())
            .append("\\\\ \\hline");
      }
    }
    sb.append("\\end{tabular}");
    return sb.toString();
  }

  /**
   * Getter for the ID field.
   * @return the ID of the node or null
   */
  public String getId() {
      return this.id.get();
    }

  /**
   * Setter for the Id field.
   * @param id id to be set
   */
  public void setId(final String id) {
    this.id.set(id);
  }

  public ConceptNode getCutoffConcept(int cutoffLevel) {
    List<ConceptNode> trace = new ArrayList<>();
    ConceptNode current = this;
    do {
      trace.add(current);
      current = current.getParent();
    } while (current.getParent() != current);
    // also add the root so that we aren't off by one when returning the cutoff level
    trace.add(current);
    return trace.get(cutoffLevel);
  }

  public String getCutoffLabel(int cutoffLevel) {
        return this.getLabel().substring(0, cutoffLevel);
  }


  public String getLabel() {
    return this.label.get();
  }

  public void setLabel(final String label) {
    this.label.set(label);
  }
}
