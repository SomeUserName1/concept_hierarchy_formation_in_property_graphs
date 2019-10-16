package kn.uni.dbis.neo4j.conceptual.algos.trestle;

import kn.uni.dbis.neo4j.conceptual.algos.cobweb.COBWEB;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptNode;
import org.neo4j.graphdb.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TRESTLE extends COBWEB {

    public TRESTLE() {
        super();
    }

    @Override
    public void integrate(Node node) {
        ConceptNode newChild = new ConceptNode();
        newChild.nodePropertiesToConcept(node);
        NodeMatcher nm = new NodeMatcher(newChild, this.root);
        ConceptNode matchedNode = nm.match();
        flatten(matchedNode);
        cobweb(matchedNode, this.root, true);
    }

    private void flatten(ConceptNode newNode) {
        // move components to top level using dot notation
        // make relations between components nominal attributes
    }

}
