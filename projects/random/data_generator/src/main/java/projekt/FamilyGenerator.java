package projekt;

import org.apache.commons.math3.analysis.function.Add;
import projekt.generators.Address;
import projekt.generators.AddressPicker;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FamilyGenerator {
	
	private PersonGenerator pg;

	public FamilyGenerator(PersonGenerator pg){
		this.pg = pg;
	}

	public void setAddress(Person person, Address address, int houseNr) {
		person.setDomicileCity(address.getCity());
		person.setDomicileStreet(address.getStreet());
		person.setHouseNumber(houseNr);
	}
	
	
	List<Person> generateFamily(){
		Random random = new Random();
		List<Person> family = new ArrayList<Person>();
		Address address = pg.pickRandomAddress();
		int houseNr = random.nextInt(35)+1;
		Person parent = pg.generateParent();
		Person spouse = pg.generateSpouse(parent);
		parent.setSpouse(spouse);
		spouse.setSpouse(parent);
		setAddress(parent, address, houseNr);
		setAddress(spouse, address, houseNr);
		family.add(parent);
		family.add(spouse);
		
		int numberOfChildren=random.nextInt(6);

		for(int i=0;i<numberOfChildren;i++) {
			Person child = pg.generateChild(parent);
			parent.addChild(child);
			spouse.addChild(child);
			child.addParent(parent);
			child.addParent(spouse);
			if(child.getAge() >= 25) {
				setAddress(child, pg.pickRandomAddress(), random.nextInt(35)+1);
			}
			else {
				setAddress(child, address, houseNr);
			}

			family.add(child);
		}


		return family;
		
	}
	
	
	
	
	
	

}
