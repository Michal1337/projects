package projekt;

import com.opencsv.bean.CsvToBeanBuilder;
import projekt.generators.StringAndWeight;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.List;
import java.util.Random;

public class Utils {
    private static Random random = new Random();

    public static <T> T pickRandom(List<T> list) {
        int randomIndex = random.nextInt(list.size());
        return list.get(randomIndex);
    }

    public static <T> List<T> importCsvFromResources(String path, Class<T> clazz) {

        var stream = Utils.class.getClassLoader().getResourceAsStream(path);

        try (InputStreamReader reader = new InputStreamReader(stream)) {
            var list = new CsvToBeanBuilder<T>(reader)
                    .withType(clazz)
                    .withSkipLines(1)
                    .withSeparator(',')
                    .build()
                    .parse();
            return list;
        } catch (IOException e) {
            e.printStackTrace();
        }

        throw new IllegalStateException("Could not load csv from: " + path);
    }

}
