package kn.uni.dbis.neo4j.conceptual.algos;

/**
 * Holder for nominal values, representing Strings, characters, booleans and IDs as String.
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NominalValue implements Value, Cloneable {
    private final String str;

    NominalValue(String value) {
        this.str = value;
    }
    NominalValue(boolean value) {
        this.str = value ? "True" : "False";
    }
    NominalValue(char value) {
        this.str = Character.toString(value);
    }
    NominalValue(long value) {this.str = Long.toHexString(value);}

    public Value clone() {
        try {
            super.clone();
        } catch (final CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return new NominalValue(this.str);
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof String) {
            String str = (String)o;
            return str.equals(this.str);
        } else if (o instanceof Boolean) {
            String bool = (Boolean) o ? "True" : "False";
            return bool.equals(this.str);
        } else if (o instanceof Character) {
            String c = Character.toString((Character)o);
            return c.equals(this.str);
        } else {
            return false;
        }
    }
}
