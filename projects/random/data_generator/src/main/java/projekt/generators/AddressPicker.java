package projekt.generators;

import projekt.Utils;

import java.util.List;

public class AddressPicker {
    private List<Address> addresses = Utils.importCsvFromResources("merged.csv", Address.class);

    public Address pickRandom() {
        return Utils.pickRandom(addresses);
    }

}
