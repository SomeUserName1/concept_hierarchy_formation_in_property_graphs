package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Objects;

/**
 * Holder for Concept values.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class ConceptValue implements Value, Cloneable {
  /**
   * The concept to encapsulate.
   */
  private final ConceptNode concept;

  /**
   * Constructor.
   *
   * @param node the node to encapsulate
   */
  ConceptValue(final ConceptNode node) {
    this.concept = node;
  }

  @Override
  public Value clone() {
    try {
      super.clone();
    } catch (final CloneNotSupportedException e) {
      e.printStackTrace();
    }
    return new ConceptValue(this.concept);
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof ConceptValue) {
      return this.concept.equals(((ConceptValue) o).concept);
    } else if (o instanceof ConceptNode) {
      return this.concept.equals(o);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.concept);
  }

  /**
   * Conditional probability of having concept c_i when given c_j bzw P(this|val).
   *
   * @param val Concept to check against (sub concept or super concept or unrelated)
   * @return The probability of this concept given the concept val
   */
  double getFactor(final ConceptValue val) {
    if (val.concept.equals(this.concept) || this.concept.isSuperConcept(val.concept)) {
      return 1.0;
    } else if (val.concept.isSuperConcept(this.concept)) {
      return (double) this.concept.getCount() / (double) val.concept.getCount();
    } else {
      // they're on different paths => disjoint
      return 0;
    }
  }
}
