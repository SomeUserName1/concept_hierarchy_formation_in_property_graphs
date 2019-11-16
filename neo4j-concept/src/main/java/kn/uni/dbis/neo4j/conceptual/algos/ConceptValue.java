package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Holder for Concept values.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class ConceptValue extends Value {
  /**
   * The concept to encapsulate.
   */
  private final AtomicReference<ConceptNode> concept;

  /**
   * Constructor.
   *
   * @param node the node to encapsulate
   */
  public ConceptValue(final ConceptNode node) {
    this.setCount(1);
    this.concept = new AtomicReference<>(node);
  }

  /**
   * Copy Constructor.
   * @param count to be set
   * @param node to be set as concept
   */
  private ConceptValue(final int count, final ConceptNode node) {
    this.setCount(count);
    this.concept = new AtomicReference<>(node);
  }

  @Override
  public Value copy() {
    return new ConceptValue(this.getCount(), this.concept.get());
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof ConceptValue) {
      return this.concept.get().equals(((ConceptValue) o).concept.get());
    } else {
      return false;
    }
  }

  @Override
  public void update(final Value other) {
    if (other instanceof ConceptValue) {
      final ConceptValue c = (ConceptValue) other;
      if (this.concept.get().equals(c.concept.get())) {
        this.setCount(this.getCount() + c.getCount());
      }
    } else {
      throw new RuntimeException("updated with wrong type!");
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.concept.get());
  }

  /**
   * Conditional probability of having concept c_i when given c_j bzw P(this|val).
   *
   * @param val Concept to check against (sub concept or super concept or unrelated)
   * @return The probability of this concept given the concept val
   */
  double getFactor(final ConceptValue val) {
    if (val.concept.get().equals(this.concept.get()) || this.concept.get().isSuperConcept(val.concept.get())) {
      return 1.0;
    } else if (val.concept.get().isSuperConcept(this.concept.get())) {
      return (double) this.concept.get().getCount() / (double) val.concept.get().getCount();
    } else {
      // they're on different paths => disjoint
      return 0;
    }
  }

  @Override
  public String toString() {
    return "ConceptValue:  count=" + this.getCount() + " Concept=("
            + this.concept.get().toString() + ")";
  }

  /**
   * Returns a string that is formatted to be used in a latex tabular environment.
   * @return a sting representation of the node for letx tables
   */
  @Override
  String toTexString() {
    return "Concept & " + this.concept.get().getLabel() + "&";
  }
}
