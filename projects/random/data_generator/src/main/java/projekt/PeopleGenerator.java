package projekt;

import projekt.generators.AddressPicker;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class PeopleGenerator {

	
	FamilyGenerator fg = new FamilyGenerator(new PersonGenerator());

	List<Group> groups = new ArrayList<Group>();
	void GeneratePeople() {
		for(int i =0; i<4; i++) {
			List<Person> people = new ArrayList<Person>();
			while(people.size()<= 65534-10)
			{
				people.addAll(fg.generateFamily());   //65534
			}
			groups.add(new Group("Group" + (i+1), people.stream().map(DisplayPerson::new).collect(Collectors.toList())));
		}
	}


	public List<Group> getPeople() {
		return groups;
	}
	

	
	
}
