package projekt;

import java.time.LocalDate;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import projekt.Person.EyeColor;
import projekt.Person.MainHand;
import projekt.Person.Sex;
import projekt.generators.Address;
import projekt.generators.AddressPicker;
import projekt.generators.WeightedStringPicker;

public class PersonGenerator {
	static WeightedStringPicker menNames = new WeightedStringPicker("7-_WYKAZ_IMION_MĘSKICH_WG_POLA_IMIĘ_PIERWSZE_WYSTĘPUJĄCYCH_W_REJESTRZE_PESEL_Z_UWZGLĘDNIENIEM_IMION_OSÓB_ZMARŁYCH.csv");
    static WeightedStringPicker menSurnames = new WeightedStringPicker("NAZWISKA_MĘSKIE-Z_UWZGLĘDNIENIEM_OSÓB_ZMARŁYCH.csv");
    static WeightedStringPicker womenNames = new WeightedStringPicker("7-_WYKAZ_IMION_ŻEŃSKICH_WG_POLA_IMIĘ_PIERWSZE_WYSTĘPUJĄCYCH_W_REJESTRZE_PESEL_Z_UWZGLĘDNIENIEM_IMION_OSÓB_ZMARŁYCH.csv");
	static WeightedStringPicker womenSurnames = new WeightedStringPicker("NAZWISKA_ŻEŃSKIE-Z_UWZGLĘDNIENIEM_OSÓB_ZMARŁYCH.csv");
    static AddressPicker addressPicker = new AddressPicker();
    Random random = new Random();
	
    @FunctionalInterface
    interface SexGenerator {
        Sex create();
    }
	
    @FunctionalInterface
    interface NameGenerator {
        String create(Sex sex);
    }

    @FunctionalInterface
    interface SurnameGenerator {
        String create(Sex sex);
    }
    
    @FunctionalInterface
    interface BirthDateGeneratorParent {
        LocalDate create();
    }
    
    @FunctionalInterface
    interface BirthDateGeneratorSpouse {
        LocalDate create(Person spouse);
    }
    
    @FunctionalInterface
    interface BirthDateGeneratorChild {
        LocalDate create(Person parent);
    }
    
    @FunctionalInterface
    interface PhoneNrGenerator {
        String create();
    }
    
    @FunctionalInterface
    interface EyeColorGenerator {
        EyeColor create(Sex s);
    }
    
    @FunctionalInterface
    interface MainHandGenerator {
    	MainHand create();
    }

    NameGenerator namegenerator = (s) -> {
        if(s.equals(Sex.Female)) {
            return womenNames.pickRandom();
        }
        else {
            return menNames.pickRandom();
        }
    };

    SurnameGenerator surnamegenerator = (s) -> {
        if(s.equals(Sex.Female)) {
            return womenSurnames.pickRandom();
        }
        else {
            return menSurnames.pickRandom();
        }
    };

     
    SexGenerator sexgenerator = () -> {
    	if(random.nextDouble()<=0.48)
    		return Sex.Male;
    	return Sex.Female;
    }; 
    
    EyeColorGenerator eyecolorgenerator = s -> {
        double prob = random.nextDouble();
        if(s.equals(Sex.Female)) {
        	if(prob<=0.9)
        		return EyeColor.Brown;
        	if(prob>0.9 && prob<=0.97)
        		return EyeColor.Blue;
        	if(prob>0.97 && prob<=0.99)
        		return EyeColor.Grey;
        	return EyeColor.Green;
        }else {
        	if(prob<=0.85)
        		return EyeColor.Brown;
        	if(prob>0.85 && prob<=0.95)
        		return EyeColor.Blue;
        	if(prob>0.95 && prob<=0.98)
        		return EyeColor.Grey;
        	return EyeColor.Green;
        }
    };
    
    MainHandGenerator mainhandgenerator = () -> {
    	if(random.nextDouble()<=0.1)
    		return MainHand.Left;
    	else
    		return MainHand.Right;
    };
        
    BirthDateGeneratorParent birthdategeneratorparent = () -> {
    	long minDay = LocalDate.of(1930, 1, 1).toEpochDay();
    	long maxDay = LocalDate.of(1990, 12, 31).toEpochDay();
    	long randomDay = ThreadLocalRandom.current().nextLong(minDay, maxDay);
    	LocalDate randomDate = LocalDate.ofEpochDay(randomDay);
    	return randomDate;
    };
    
    BirthDateGeneratorChild birthdategeneratorchild = (parent) -> {
    	long minDay = parent.getBirthDate().plusYears(20).toEpochDay();
    	long maxDay = parent.getBirthDate().plusYears(31).toEpochDay();
    	long randomDay = ThreadLocalRandom.current().nextLong(minDay, maxDay);
    	LocalDate randomDate = LocalDate.ofEpochDay(randomDay);
    	return randomDate;
    };
    
    BirthDateGeneratorSpouse birthdategeneratorspouse = (spouse) -> {
    	long minDay = spouse.getBirthDate().minusYears(5).toEpochDay();
    	long maxDay = spouse.getBirthDate().plusYears(5).toEpochDay();
    	long randomDay = ThreadLocalRandom.current().nextLong(minDay, maxDay);
    	LocalDate randomDate = LocalDate.ofEpochDay(randomDay);
    	return randomDate;
    };
    
    	
    PhoneNrGenerator phonenrgenerator = () -> {
    	int phonenr=random.nextInt(900000000)+100000000;
    	String strphonenr = String.valueOf(phonenr);
    	return strphonenr;
    	};

    public Address pickRandomAddress() {
        return addressPicker.pickRandom();
    }


    Person generateParent(){
    	Sex s = Sex.Male;
    	LocalDate birthdate = birthdategeneratorparent.create();

    	
    	Person parent = new Person(namegenerator.create(s),surnamegenerator.create(s),s,birthdate,phonenrgenerator.create(), new Pesel(birthdate, s),
    			new IdentityCard(),mainhandgenerator.create(),eyecolorgenerator.create(s));

    	return parent;

    }

    Person generateSpouse(Person spouse){
    	Sex s = Sex.Female;
    	LocalDate birthdate = birthdategeneratorspouse.create(spouse);
    	
    	Person parent = new Person(namegenerator.create(s),surnamegenerator.create(s),s,birthdate,phonenrgenerator.create(), new Pesel(birthdate, s),
    			new IdentityCard(),mainhandgenerator.create(),eyecolorgenerator.create(s));

    	return parent;

    }


    Person generateChild(Person parent) {
    	Sex s = sexgenerator.create();
    	LocalDate birthdate = birthdategeneratorchild.create(parent);
    	Person child = null;
    	if((LocalDate.now().getYear()-birthdate.getYear())>=18) {
    		 child = new Person(namegenerator.create(s), parent.getSurname(),s,birthdate,phonenrgenerator.create(), new Pesel(birthdate, s),
        			new IdentityCard(),mainhandgenerator.create(),eyecolorgenerator.create(s));
    	}else {
    		 child = new Person(namegenerator.create(s), parent.getSurname(),s,birthdate,phonenrgenerator.create(), new Pesel(birthdate, s),
        			mainhandgenerator.create(),eyecolorgenerator.create(s));
    	}

    	return child;
    }
    
    
    
}
