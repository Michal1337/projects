package projekt.generators;

import com.opencsv.bean.CsvBindByPosition;
import lombok.*;

@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
public class Address {
    @CsvBindByPosition(position = 1)
    private String street;

    @CsvBindByPosition(position = 2)
    private String city;
}
