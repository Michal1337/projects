package projekt.generators;

import com.opencsv.bean.CsvToBeanBuilder;
import projekt.Utils;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Random;

public class WeightedStringPicker {
    private List<StringAndWeight> list;
    private int totalWeight;

    public WeightedStringPicker(String path) {
        list = Utils.importCsvFromResources(path, StringAndWeight.class);
        totalWeight = list.stream().mapToInt(StringAndWeight::getWeight).sum();
    }

    public String pickRandom() {
        var random = new Random();
        int num = random.nextInt(totalWeight);

        int i = 0;
        while(num > list.get(i).getWeight()) {
            num -= list.get(i).getWeight();
            i++;
        }

        return list.get(i).getString();
    }
}
