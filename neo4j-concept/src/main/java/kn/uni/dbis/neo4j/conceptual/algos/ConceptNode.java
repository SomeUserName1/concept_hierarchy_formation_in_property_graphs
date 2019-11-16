package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.PropertyContainer;
import org.neo4j.graphdb.Relationship;

/**
 * The basic data type for the conceptual hierarchy (tree) constructed and used by cobweb.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class ConceptNode {
  /**
   * Attribute aggregation over all (sub-)instances of this Concept.
   */
  private final ConcurrentMap<String, List<Value>> attributes = new ConcurrentHashMap<>();
  /**
   * Number of Nodes under this ConceptNode.
   */
  private final AtomicInteger count = new AtomicInteger(1);
  /**
   * Neo4j ID of the Incorporated node..
   */
  private final AtomicReference<String> id = new AtomicReference<>(null);

  /**
   * Label of the node.
   */
  private final AtomicReference<String> label = new AtomicReference<>(null);

  /**
   * The sub-concepts of this Concept.
   */
  private final List<ConceptNode> children = Collections.synchronizedList(new ArrayList<>());
  /**
   * The super-concept of this concept.
   */
  private final AtomicReference<ConceptNode> parent = new AtomicReference<>(null);

  /**
   * Constructor.
   */
  public ConceptNode() {
  }

  /**
   * Copy constructor.
   *
   * @param node ConceptNode to copy.
   */
  public ConceptNode(final ConceptNode node) {
    List<Value> values;
    String attributeName;
    for (ConcurrentMap.Entry<String, List<Value>> attribute : node.getAttributes().entrySet()) {
      attributeName = attribute.getKey();
      values = Collections.synchronizedList(new ArrayList<>());
      this.attributes.put(attributeName, values);
      synchronized (attribute.getValue()) {
        for (Value value : attribute.getValue()) {
          synchronized (values) {
            values.add(value.copy());
          }
        }
      }
    }

    synchronized (this.children) {
      this.children.addAll(node.getChildren());
    }

    this.setParent(node.getParent());
  }

  /**
   * Converts a property container into a Singleton ConceptNode.
   * Orders are ignored, so all lists are converted to sets.
   *
   * @param propertyContainer The property container to parse.
   */
  ConceptNode(final PropertyContainer propertyContainer) {

    List<Value> values = Collections.synchronizedList(new ArrayList<>());

    if (propertyContainer instanceof Relationship) {
      final Relationship rel = (Relationship) propertyContainer;
      this.setId(Long.toString(rel.getId()));
      synchronized (values) {
        values.add(new NominalValue(rel.getType().name()));
      }
      this.attributes.put("RelType", values);
    } else if (propertyContainer instanceof Node) {
      final Node mNode = (Node) propertyContainer;
      this.setId(Long.toString(mNode.getId()));
      for (Label label : mNode.getLabels()) {
        synchronized (values) {
          values.add(new NominalValue(label.name()));
        }
        this.attributes.put("Label", values);
      }
    }

    // loop over the properties of a N4J node and cast them to a Value
    Object o;
    Object[] arr;
    for (ConcurrentMap.Entry<String, Object> property : propertyContainer.getAllProperties().entrySet()) {
      values = Collections.synchronizedList(new ArrayList<>());
      o = property.getValue();
      if (o.getClass().isArray()) {
        arr = (Object[]) o;

        for (Object ob : arr) {
          synchronized (values) {
            values.add(Value.cast(ob));
          }
        }
      } else {
        synchronized (values) {
          values.add(Value.cast(property.getValue()));
        }
      }
      this.attributes.put(property.getKey(), values);
    }
  }

  /**
   * Sets the fields of a ConceptNode to be appropriate for a root.
   * @return the transformed node
   */
  public ConceptNode root() {
    this.setCount(0);
    this.setParent(this);

    return this;
  }

  /**
   * Aggregates the count and attributes of the nodes, hosted by this concept.
   *
   * @param usedForUpdate  the node to incorporate into the concept.
   */
  public void updateCounts(final ConceptNode usedForUpdate) {
    List<Value> thisValues;
    NumericValue thisNumeric;
    boolean matched;

    this.setCount(this.getCount() + usedForUpdate.getCount());
    // loop over the attributes of the node to incorporate
    for (ConcurrentMap.Entry<String, List<Value>> otherAttributes : usedForUpdate.getAttributes().entrySet()) {

        thisValues = this.getAttributes().get(otherAttributes.getKey());
        // If the attribute is present in this node, check against the present values
        if (thisValues != null) {
          synchronized (this.getAttributes().get(otherAttributes.getKey())) {
          synchronized (otherAttributes.getValue()) {
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
          }
          }
        } else {
          synchronized (this.getAttributes().get(otherAttributes.getKey())) {
            synchronized (otherAttributes.getValue()) {
              // Else add the attribute and it's properties of the new node as they are
              final List<Value> copies = Collections.synchronizedList(new ArrayList<>());
              for (Value value : otherAttributes.getValue()) {
                copies.add(value.copy());
              }
              this.getAttributes().put(otherAttributes.getKey(), copies);
            }}
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
    if (this.getParent() == this) {
      return false;
    } else if (this.getParent().equals(c)) {
      return true;
    } else {
      return this.getParent().isSuperConcept(c);
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
  private void setCount(final int count) {
    this.count.set(count);
  }

  /**
   * Getter for the Attribute Value aggregation map.
   *
   * @return the map that stores the attributes as strings and possible values with counts as map
   */
  public ConcurrentMap<String, List<Value>> getAttributes() {
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
    synchronized (this.children) {
      this.children.add(node);
    }
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


  @Override
  public boolean equals(final Object o) {
    if (o instanceof ConceptNode) {
      final ConceptNode node = (ConceptNode) o;
      return node.getCount() == this.getCount() && node.getAttributes().equals(this.getAttributes());
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.getCount(), this.getAttributes());
  }

  @Override
  public String toString() {
    final String id = this.getId() != null ? "ID: " + this.getId() : "";
    return "ConceptNode " + id + " Count: " + this.getCount() + " Attributes: "
        + this.getAttributes();
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

  /**
   * Returns the label of the superclass that is cutoffLevel steps away from the root.
   * @param cutoffLevel how far the returned concept should be from the root
   * @return the label of a super-concept of the current node
   */
  String getCutoffLabel(final int cutoffLevel) {
        return this.getLabel().substring(0, cutoffLevel);
  }

  /**
   * getter for the label.
   * @return the label of this node
   */
  String getLabel() {
    return this.label.get();
  }

  /**
   * setter for the label.
   * @param label the label to be set in this node
   */
  void setLabel(final String label) {
    this.label.set(label);
  }

  /**
   * setter for children.
   * @param idx index
   * @param child to be set
   */
  void setChild(final int idx, final ConceptNode child) {
    synchronized (this.children) {
      this.children.set(idx, child);
    }
  }

  /**
   * removes children.
   * @param child to be removed
   */
  void removeChild(final ConceptNode child) {
    synchronized (this.children) {
      this.children.remove(child);
    }
  }

  /**
   *
   * clears the children.
   */
  void clearChildren() {
    synchronized (this.children) {
      this.children.clear();
    }
  }
 /*
   * Returns the super-concept of the node cutoffLevel traversal steps away from the root.
   * @param cutoffLevel how far the returned concept should be from the root
   * @return A super-concept of the current node

  public ConceptNode getCutoffConcept(final int cutoffLevel) {
    final List<ConceptNode> trace = new ArrayList<>();
    ConceptNode current = this;
    do {
      trace.add(current);
      current = current.getParent();
    } while (current.getParent() != current);
    // also add the root so that we aren't off by one when returning the cutoff level
    trace.add(current);
    return trace.get(cutoffLevel);
  }*/
}
