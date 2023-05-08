package projekt;

import org.jxls.common.Context;
import org.jxls.util.JxlsHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TestPlikow {

    private static Logger logger = LoggerFactory.getLogger(TestPlikow.class);

    public class testZiomek{
        private String imie;
        public String nazwisko;
        public int wiek;

        public String getImie() {
            return "debson";
        }

        public testZiomek(String imie, String nazwisko, int wiek) {
            this.imie = imie;
            this.nazwisko = nazwisko;
            this.wiek = wiek;
        }
    }

    public void tescik(){
        logger.info("Running Object Collection demo");
        testZiomek o1 = new testZiomek("Daniel", "Tyt",13);
        testZiomek o2 = new testZiomek("Majkel", "Grom",16);
        List<testZiomek> lista = new ArrayList<>();
        Random random = new Random();
//        for(int i = 0; i<100_000;i++ ){
//            o1 = new testZiomek("Szrek","danielek", random.nextInt(100)+20);
//            lista.add(o1);
//        }
        lista.add(o1);
        lista.add(o2);
        try(InputStream is = new FileInputStream("C:\\Users\\danie\\Documents\\template.xls")) {
            try (OutputStream os = new FileOutputStream("C:\\Users\\danie\\Documents\\docelowy.xls")) {
                Context context = new Context();
                context.putVar("testZiomeks", lista);
                JxlsHelper.getInstance().processTemplate(is, os, context);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


    }


}
