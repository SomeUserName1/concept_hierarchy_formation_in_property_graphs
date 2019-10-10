package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

public class NominalValue implements Value, Cloneable {
    private String str;

    NominalValue(String value) {
        this.str = value;
    }
    NominalValue(boolean value) {
        this.str = value ? "True" : "False";
    }
    NominalValue(char value) {
        this.str = Character.toString(value);
    }

    public Value clone() {
        try {
            super.clone();
        } catch (final CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return new NominalValue(this.str);
    }

    public String getStr() {
        return this.str;
    }

    public void setStr(String str) {
        this.str = str;
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
