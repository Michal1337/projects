package projekt.generators;

import com.opencsv.bean.CsvBindByPosition;

public class StringAndWeight {
    @CsvBindByPosition(position = 0)
    private String string;

    @CsvBindByPosition(position = 1)
    private int weight;

    public String getString() {
        return string;
    }

    public int getWeight() {
        return weight;
    }

    public void setString(String string) {
        this.string = string;
    }

    public void setWeight(int weight) {
        this.weight = weight;
    }
}
