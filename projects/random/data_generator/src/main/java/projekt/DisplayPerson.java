package projekt;

import java.util.List;
import java.util.stream.Collectors;

public class DisplayPerson {
    private Person person;

    public DisplayPerson(Person person) {
        this.person = person;
    }

    public Person.EyeColor getEyecolor() {
        return person.getEyecolor();
    }

    public String getChildren() {
        return convertToPeselList(person.getChildren());
    }

    public String getParents() {
        return convertToPeselList(person.getParents());
    }

    private String convertToPeselList(List<Person> list){
        String t = list
                .stream()
                .map(s -> s.getPesel().pesel)
                .collect(Collectors.joining(", "));
        return t;
    }

    public String getSpouse() {
        return person.getSpouse().getPesel().pesel;
    }

    public Person.MainHand getMainhand() {
        return person.getMainhand();
    }

    public Pesel getPesel() {
        return person.getPesel();
    }

    public String getIdentityCardNumber() {
        return person.getIdentityCard().getIdentityCardNumber();
    }

    public String getPhoneNr() {
        return person.getPhoneNr();
    }

    public String getDomicileStreet() {
        return person.getDomicileStreet();
    }

    public int getHouseNumber() {
        return person.getHouseNumber();
    }

    public String getDomicileCity() {
        return person.getDomicileCity();
    }

    public int getAge() {
        return person.getAge();
    }

    public String getBirthDate() {
        return person.getBirthDate().toString();
    }

    public Person.Sex getSex() {
        return person.getSex();
    }

    public String getSurname() {
        return person.getSurname();
    }

    public String getName() {
        return person.getName();
    }
}
