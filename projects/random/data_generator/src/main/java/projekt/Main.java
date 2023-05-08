package projekt;

import projekt.xls.XlsCreator;

import java.time.LocalDate;

public class Main {

	public static void main(String[] args) {
		PeopleGenerator pg = new PeopleGenerator();
		pg.GeneratePeople();
		XlsCreator.createXls(pg.getPeople(), "docelowy.xls");

	}

}
