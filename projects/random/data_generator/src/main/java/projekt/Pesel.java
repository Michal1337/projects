package projekt;

import java.time.LocalDate;
import java.time.Month;
import java.util.List;

import projekt.Person.Sex;

public class Pesel {
	private static final LocalDate JAN_1ST_2000 = LocalDate.of(2000, Month.JANUARY, 1);

	String pesel = "";
	static int counter = 0;

	public Pesel(LocalDate birthDate, Sex sex){
		pesel = generatePesel(birthDate, sex);
	}

	private String generatePesel(LocalDate birthDate, Sex sex) {
		var stringBuilder = new StringBuilder();

		appendBirthDate(birthDate, stringBuilder);
		appendCounter(stringBuilder);
		appendSex(sex, stringBuilder);
		appendChecksum(stringBuilder);

		return stringBuilder.toString();
	}

	private void appendBirthDate(LocalDate birthDate, StringBuilder stringBuilder) {
		int year = birthDate.getYear() % 100;

		if(year <= 9) {
			stringBuilder.append('0');
		}

		stringBuilder.append(year);

		if(birthDate.isBefore(JAN_1ST_2000)) {
			int month = birthDate.getMonthValue();
			if(month <= 9) {
				stringBuilder.append("0");
			}
			stringBuilder.append(birthDate.getMonthValue());
		} else {
			stringBuilder.append(birthDate.getMonthValue() + 20);
		}
		int day = birthDate.getDayOfMonth();
		if(day <= 9) {
			stringBuilder.append("0");
		}
		stringBuilder.append(birthDate.getDayOfMonth());
	}

	private void appendCounter(StringBuilder stringBuilder) {
		int count = counter % 1000;

		var string = String.format("%03d", count);

		stringBuilder.append(string);
		counter++;
	}

	private void appendSex(Sex sex, StringBuilder stringBuilder) {
		var femaleDigits = List.of(0, 2, 4, 6, 8);
		var maleDigits = List.of(1, 3, 5, 7, 9);

		if(sex.equals(Sex.Female)) {
			stringBuilder.append(Utils.pickRandom(femaleDigits));
		} else {
			stringBuilder.append(Utils.pickRandom(maleDigits));
		}
	}

	private void appendChecksum(StringBuilder stringBuilder) {
		int SumWithWeights = 0;

		int[] Weights = new int[] { 3,1,9,7,3,1,9,7,3,1 };
		Long tmp = Long.parseLong(stringBuilder.toString());
		int i = 0;

		while(tmp != 0) {
			SumWithWeights+=Weights[i] * tmp % 10;
			tmp /= 10;
			i++;
		}

		if(SumWithWeights % 10 == 0) {
			stringBuilder.append(0);
		} else {
			stringBuilder.append(10 - SumWithWeights % 10);
		}
	}

	@Override
	public String toString() {
		return pesel;
	}
}
