package projekt;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.apache.log4j.chainsaw.Main;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Getter
@Setter
@AllArgsConstructor
public class Person {
	private String name;
	private String surname;
	private Sex sex;
	private LocalDate birthDate;
	private int age;
	private String domicileCity;
	private String domicileStreet;
	private int houseNumber;
	private String phoneNr;
	private IdentityCard identityCard;
	private Pesel pesel;

	private MainHand mainhand;
	private EyeColor eyecolor;

	private Person spouse;
	
	private List<Person> children = new ArrayList<>();
	private List<Person> parents = new ArrayList<>();

	public Person(String name, String surname, Sex sex, LocalDate birthDate, String domicileCity, String domicileStreet, int houseNumber, String phoneNr, IdentityCard identityCard, Pesel pesel, MainHand mainhand, EyeColor eyecolor, Person spouse, List<Person> children, List<Person> parents) {
		this.name = name;
		this.surname = surname;
		this.sex = sex;
		this.birthDate = birthDate;
		this.domicileCity = domicileCity;
		this.domicileStreet = domicileStreet;
		this.houseNumber = houseNumber;
		this.phoneNr = phoneNr;
		this.identityCard = identityCard;
		this.pesel = pesel;
		this.mainhand = mainhand;
		this.eyecolor = eyecolor;
		this.spouse = spouse;
		this.children = children;
		this.parents = parents;
	}

	public Person(String name, String surname, Sex sex, LocalDate birthDate, String phoneNr, Pesel pesel, IdentityCard identitycard, MainHand mainhand, EyeColor eyecolor) {
		this.name=name;
		this.surname=surname;
		this.sex=sex;
		this.birthDate=birthDate;
		this.phoneNr=phoneNr;
		this.pesel=pesel;
		this.identityCard=identitycard;
		this.mainhand=mainhand;
		this.eyecolor=eyecolor;
		this.age = LocalDate.now().getYear() - birthDate.getYear();
	}

	public Person(String name, String surname, Sex sex, LocalDate birthDate, String phoneNr, Pesel pesel, MainHand mainhand, EyeColor eyecolor) {
		this.name=name;
		this.surname=surname;
		this.sex=sex;
		this.birthDate=birthDate;
		this.phoneNr=phoneNr;
		this.pesel=pesel;
		this.mainhand=mainhand;
		this.eyecolor=eyecolor;
		this.age = LocalDate.now().getYear() - birthDate.getYear();
	}

	public void addChild(Person child) {
		this.children.add(child);
	}

	public void addParent(Person parent) {
		this.parents.add(parent);
	}

	public enum Sex {
	 	Male, Female
	}

	public enum MainHand {
	 	Left, Right
	}

	public enum EyeColor{
		Green, Blue, Brown, Grey
	}
}


