package kn.uni.dbis.neo4j.conceptual.algos.trestle;

import kn.uni.dbis.neo4j.conceptual.algos.cobweb.COBWEB;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptNode;
import org.neo4j.graphdb.Node;

import java.util.ArrayList;
import java.util.List;

public class TRESTLE extends COBWEB {

    public TRESTLE() {
        super();
    }

    @Override
    public void integrate(Node node) {
        ConceptNode newChild = new ConceptNode();
        newChild.propertyContainerToConceptNode(node);
        match(newChild);
        flatten(newChild);
        cobweb(newChild, this.root, true);
    }

    private void match(ConceptNode toMatch) {
        List<String> rootAttribs = new ArrayList<>(root.getAttributes().keySet());
        List<String> toMatchAttrib = new ArrayList<>(toMatch.getAttributes().keySet());

        double baseEAP = getExpectedAttributePrediction(toMatch);
        double[][] costMatrix = new double[toMatchAttrib.size()][rootAttribs.size()];
        ConceptNode altered;
        int i = 0, j = 0;
        double min;
        int[] minIdx = new int[toMatchAttrib.size()];
        for (String toMatchName : toMatchAttrib) {
            min = 1;
            for (String rootName : rootAttribs) {
                altered = toMatch.clone();
                altered.getAttributes().put(rootName, altered.getAttributes().get(toMatchName));
                altered.getAttributes().remove(toMatchName);
                costMatrix[j][i] = 1 - (COBWEB.getExpectedAttributePrediction(altered) - baseEAP);
                if (costMatrix[j][i] < min) {
                    min = costMatrix[j][i];
                    minIdx[j] = i;
                }
                i++;
            }
            j++;
        }

        // Transform node
        for (j = 0; j < costMatrix.length; j++) {
            if (costMatrix[j][minIdx[j]] < 1) {
                toMatch.getAttributes().put(rootAttribs.get(minIdx[j]), toMatch.getAttributes().get(toMatchAttrib.get(j)));
                toMatch.getAttributes().remove(toMatchAttrib.get(j));
            }
        }
    }

    private void flatten(ConceptNode newNode) {
        // move components to top level using dot notation
        // make relations between components nominal attributes
        // TODO
    }

}
